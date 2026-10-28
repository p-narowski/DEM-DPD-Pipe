"""Per-step Python snapshots; these are NOT integrator-stage instrumentation.
Serial/OpenMP only. Read-only. Effective integration mass is deliberately NaN.
"""
import csv
import math
import KratosMultiphysics as KM
import KratosMultiphysics.DEMApplication as DEM

class ParticleStepLogger:
    def __init__(self, model_part, start_time=5.6, end_time=6.0,
                 filename="particle_step_debug.csv"):
        self.mp = model_part
        self.start, self.end = start_time, end_time
        if model_part.GetCommunicator().GetDataCommunicator().Size() != 1:
            raise RuntimeError("Serial/OpenMP only")
        self.before = None
        self.serial = 0
        self.last_time = None
        fields = ["sample", "time", "dt", "node_id", "particle_type",
                  "mass_before", "mass_after", "effective_integrator_mass",
                  "mass_unchanged", "force_stage_verified"]
        self.prefixes = ["position_before", "position_after", "velocity_before",
                         "velocity_after", "total_force_before_snapshot",
                         "total_force_after_snapshot", "coupling_after_snapshot",
                         "external_after_snapshot", "delta_p", "observed_rate",
                         "rate_minus_total_snapshot", "rate_minus_coupling_external",
                         "response_ratio", "apparent_mass_from_total_snapshot"]
        fields += [p+"_"+a for p in self.prefixes for a in "xyz"]
        self.file = open(filename, "w", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()

    @staticmethod
    def read(node, var):
        if var is None:
            return None
        if node.SolutionStepsDataHas(var):
            return node.GetSolutionStepValue(var)
        if node.Has(var):
            return node.GetValue(var)
        return None

    def vector(self, node, var):
        value = self.read(node,var)
        return [float(value[i]) for i in range(3)] if value is not None else [math.nan]*3

    def snapshot(self):
        result = {}
        for n in self.mp.Nodes:
            mass = self.read(n, KM.NODAL_MASS)
            if mass is None:
                raise RuntimeError("Missing tracker mass on node {}".format(n.Id))
            dem = n.Is(DEM.IS_SUSPENDED_PARTICLE)
            cv = getattr(DEM, "DPD_TO_DEM_COUPLING_FORCE" if dem else
                         "DEM_TO_DPD_COUPLING_FORCE", None)
            result[n.Id] = dict(m=float(mass), kind="DEM" if dem else "DPD",
                position=[n.X,n.Y,n.Z], velocity=self.vector(n,KM.VELOCITY),
                total=self.vector(n,KM.TOTAL_FORCES), coupling=self.vector(n,cv),
                external=self.vector(n,KM.EXTERNAL_APPLIED_FORCE))
        return result

    def BeginStep(self):
        time = float(self.mp.ProcessInfo[KM.TIME])
        self.before = None
        if self.start-1e-12 <= time <= self.end+1e-12:
            self.before = (time, float(self.mp.ProcessInfo[KM.DELTA_TIME]), self.snapshot())

    def EndStep(self):
        if self.before is None:
            return
        time, dt, old = self.before
        self.before = None
        if time == self.last_time:
            return
        if dt <= 0 or float(self.mp.ProcessInfo[KM.TIME]) != time:
            raise RuntimeError("Invalid step pairing")
        new = self.snapshot()
        if old.keys() != new.keys():
            raise RuntimeError("Particle population changed inside diagnostic step")
        self.serial += 1
        for ident, b in old.items():
            a = new[ident]
            dp = [a['m']*a['velocity'][i]-b['m']*b['velocity'][i] for i in range(3)]
            rate = [v/dt for v in dp]
            values = dict(position_before=b['position'], position_after=a['position'],
                velocity_before=b['velocity'], velocity_after=a['velocity'],
                total_force_before_snapshot=b['total'], total_force_after_snapshot=a['total'],
                coupling_after_snapshot=a['coupling'], external_after_snapshot=a['external'],
                delta_p=dp, observed_rate=rate,
                rate_minus_total_snapshot=[rate[i]-a['total'][i] for i in range(3)],
                rate_minus_coupling_external=[rate[i]-a['coupling'][i]-a['external'][i] for i in range(3)],
                response_ratio=[rate[i]/a['total'][i] if abs(a['total'][i])>1e-20 else math.nan for i in range(3)],
                apparent_mass_from_total_snapshot=[a['total'][i]*dt/(a['velocity'][i]-b['velocity'][i])
                    if abs(a['velocity'][i]-b['velocity'][i])>1e-20 else math.nan for i in range(3)])
            row = dict(sample=self.serial,time=time,dt=dt,node_id=ident,particle_type=a['kind'],
                mass_before=b['m'],mass_after=a['m'],effective_integrator_mass=math.nan,
                mass_unchanged=int(a['m']==b['m']),force_stage_verified=0)
            for prefix, value in values.items():
                for i,axis in enumerate('xyz'):
                    row[prefix+'_'+axis] = value[i]
            self.writer.writerow(row)
        self.file.flush()
        self.last_time = time

    def close(self):
        if not self.file.closed:
            self.file.close()