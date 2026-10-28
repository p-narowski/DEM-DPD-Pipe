"""Serial/OpenMP DEM-DPD momentum and impulse diagnostics.

Call Execute() after EVERY completed solver step, including one initial call
before the first step. CSV writing is throttled internally. Forces sampled at
step completion must be the forces actually used for that velocity update.
No MPI/global reduction is implemented. No solver state is modified.
"""
import csv
import math
import os
import KratosMultiphysics as KM
import KratosMultiphysics.DEMApplication as DEM


def vec(value):
    result = [float(value[i]) for i in range(3)]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("Non-finite vector in momentum diagnostic")
    return result


def add(a, b):
    return [a[i] + b[i] for i in range(3)]


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def scale(a, factor):
    return [factor * x for x in a]


def norm(a):
    return math.hypot(*a)


class MomentumTracker:
    """Track P and integral(F dt), with explicit completeness flags.

    wall_mode:
      'none': caller asserts that no wall force exists.
      'callback': wall_force_provider(model_part) returns the net wall force
                  ON PARTICLES, evaluated at the force stage used by solver.
      'unknown': wall force and physical balance residuals are NaN.

    Optional force providers return three-component SYSTEM TOTALS:
      external_force_provider replaces the default sum of
      EXTERNAL_APPLIED_FORCE over ALL particles plus sum(m)*gravity.
      other_force_provider supplies other external forces, e.g. constraints.
      wall_force_provider supplies ONLY wall force, not TOTAL_FORCES.

    Default gravity is explicitly zero. Set it for gravitational cases.
    forces_complete=True is the caller's assertion that these categories
    cover ALL external forces and use the correct integration-stage values.
    Keep it False until this has been verified.

    A changed ID set or mass invalidates the physical residual for that CSV
    interval. This deliberately does not estimate inlet/deletion impulses.
    Start a clean interval with ResetBaseline() after inlet activity ends.
    Stable IDs do not detect imposed velocities: those require accounting
    for their impulses or forces_complete=False.
    """

    VECTOR_FIELDS = (
        'P_dem', 'P_dpd', 'P_total', 'F_external', 'F_wall', 'F_other',
        'F_expected', 'F_dpd_to_dem', 'F_dem_to_dpd',
        'F_coupling_residual', 'dP_total_dt', 'J_external', 'J_wall',
        'J_other', 'J_expected', 'delta_P', 'impulse_balance_error',
        'momentum_balance_error',
    )

    def __init__(self, particles_model_part, output_directory='.',
                 suspended_radius_threshold=0.19,
                 output_file_name='momentum_history.csv', append=False,
                 output_interval=0.1, classification='flag',
                 wall_mode='unknown', wall_force_provider=None,
                 external_force_provider=None, other_force_provider=None,
                 gravity=(0.0, 0.0, 0.0), forces_complete=False):
        if append:
            raise ValueError('Appending without restoring impulse state is unsupported')
        if classification not in ('flag', 'radius'):
            raise ValueError("classification must be 'flag' or 'radius'")
        if wall_mode not in ('none', 'unknown', 'callback'):
            raise ValueError('Invalid wall_mode')
        if wall_mode == 'callback' and wall_force_provider is None:
            raise ValueError('wall_force_provider is required')
        if wall_mode != 'callback' and wall_force_provider is not None:
            raise ValueError("Use wall_mode='callback' with a wall provider")
        if float(output_interval) < 0:
            raise ValueError('output_interval must be nonnegative')
        self.mp = particles_model_part
        self.threshold = float(suspended_radius_threshold)
        self.classification = classification
        self.wall_mode = wall_mode
        self.wall_provider = wall_force_provider
        self.external_provider = external_force_provider
        self.other_provider = other_force_provider
        self.gravity = vec(gravity)
        self.forces_complete = bool(forces_complete)
        self.output_interval = float(output_interval)
        communicator = self.mp.GetCommunicator().GetDataCommunicator()
        if communicator.Size() != 1:
            raise RuntimeError('This tracker supports serial/OpenMP only, not MPI')
        os.makedirs(output_directory or '.', exist_ok=True)
        self.output_file_name = os.path.join(output_directory or '.', output_file_name)
        fields = ['step', 'time', 'dt', 'solver_dt', 'n_particles',
                  'population_unchanged', 'every_step_sampled',
                  'force_budget_complete', 'balance_valid', 'coupling_available',
                  'F_coupling_residual_norm', 'impulse_balance_error_norm',
                  'momentum_balance_error_norm']
        fields += [prefix + '_' + axis for prefix in self.VECTOR_FIELDS for axis in 'xyz']
        self._file = open(self.output_file_name, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=fields)
        self._writer.writeheader()
        self._last_time = None
        self._last_step = None
        self._population = None
        self._last_write_time = None
        self._last_write_p = None
        self._latest = None
        self._clear_interval()

    @staticmethod
    def _read(node, variable, default=None):
        if node.SolutionStepsDataHas(variable):
            return node.GetSolutionStepValue(variable)
        if node.Has(variable):
            return node.GetValue(variable)
        if default is not None:
            return default
        raise RuntimeError('Missing {} on node {}'.format(variable.Name(), node.Id))

    def _is_dem(self, node):
        if self.classification == 'radius':
            return float(self._read(node, KM.RADIUS)) >= self.threshold
        return node.Is(DEM.IS_SUSPENDED_PARTICLE)

    def _snapshot(self):
        pd, pf = [0.0]*3, [0.0]*3
        ext_terms = [[], [], []]
        dem_terms = [[], [], []]
        dpd_terms = [[], [], []]
        coupling = [[], []]
        coupling_available = True
        population = {}
        cv_dem = getattr(DEM, 'DPD_TO_DEM_COUPLING_FORCE', None)
        cv_dpd = getattr(DEM, 'DEM_TO_DPD_COUPLING_FORCE', None)
        for node in self.mp.Nodes:
            mass = float(self._read(node, KM.NODAL_MASS))
            if not math.isfinite(mass) or mass < 0:
                raise ValueError('Invalid NODAL_MASS on node {}'.format(node.Id))
            population[node.Id] = mass
            velocity = vec(self._read(node, KM.VELOCITY))
            is_dem = self._is_dem(node)
            terms = dem_terms if is_dem else dpd_terms
            applied = vec(self._read(node, KM.EXTERNAL_APPLIED_FORCE, [0.0]*3))
            for i in range(3):
                terms[i].append(mass * velocity[i])
                ext_terms[i].append(applied[i] + mass * self.gravity[i])
            variable = cv_dem if is_dem else cv_dpd
            if variable is None:
                coupling_available = False
            elif node.SolutionStepsDataHas(variable) or node.Has(variable):
                coupling[0 if is_dem else 1].append(vec(self._read(node, variable)))
            else:
                coupling_available = False
        pd = [math.fsum(t) for t in dem_terms]
        pf = [math.fsum(t) for t in dpd_terms]
        ext = (vec(self.external_provider(self.mp)) if self.external_provider
               else [math.fsum(t) for t in ext_terms])
        wall = ([0.0]*3 if self.wall_mode == 'none' else
                vec(self.wall_provider(self.mp)) if self.wall_mode == 'callback'
                else [float('nan')]*3)
        other = vec(self.other_provider(self.mp)) if self.other_provider else [0.0]*3
        if coupling_available:
            cd, cf = [[math.fsum(v[i] for v in group) for i in range(3)]
                      for group in coupling]
        else:
            cd, cf = [float('nan')]*3, [float('nan')]*3
        return dict(P_dem=pd, P_dpd=pf, P_total=add(pd, pf),
                    F_external=ext, F_wall=wall, F_other=other,
                    F_expected=add(add(ext, wall), other),
                    F_dpd_to_dem=cd, F_dem_to_dpd=cf,
                    F_coupling_residual=add(cd, cf),
                    coupling_available=coupling_available,
                    population=population)

    def _clear_interval(self):
        self._impulses = {name: [0.0]*3 for name in ('external', 'wall', 'other')}
        self._population_ok = True
        self._sampling_ok = True

    def Execute(self):
        """Sample every completed step; return False for duplicate-time calls."""
        time = float(self.mp.ProcessInfo[KM.TIME])
        step = int(self.mp.ProcessInfo[KM.STEP])
        if self._last_time is not None:
            if time == self._last_time:
                return False
            if time < self._last_time:
                raise RuntimeError('Simulation time moved backwards')
        snapshot = self._snapshot()
        solver_dt = float(self.mp.ProcessInfo[KM.DELTA_TIME])
        if self._last_time is None:
            self._last_write_time = time
            self._last_write_p = snapshot['P_total'][:]
        else:
            dt = time - self._last_time
            dt_matches = math.isclose(
                dt,
                solver_dt,
                rel_tol=1e-7,
                abs_tol=1e-12,
            )

            self._sampling_ok = self._sampling_ok and dt_matches
            self._population_ok = (self._population_ok and
                                   snapshot['population'] == self._population)
            for name in self._impulses:
                self._impulses[name] = add(self._impulses[name],
                                          scale(snapshot['F_' + name], dt))
        self._last_time, self._last_step = time, step
        self._population = snapshot['population']
        self._latest = (time, step, solver_dt, snapshot)
        if time == self._last_write_time and self._last_write_p == snapshot['P_total']:
            # Only the initial sample has an empty interval here.
            if not getattr(self, '_initial_written', False):
                self._write()
                self._initial_written = True
        elif time - self._last_write_time >= self.output_interval - 1e-12:
            self._write()
        return True

    def _write(self):
        time, step, solver_dt, snapshot = self._latest
        dt = time - self._last_write_time
        nan = [float('nan')]*3
        delta = sub(snapshot['P_total'], self._last_write_p)
        j = add(add(self._impulses['external'], self._impulses['wall']),
                self._impulses['other'])
        complete = self.forces_complete and self.wall_mode != 'unknown'
        valid = dt > 0 and complete and self._sampling_ok and self._population_ok
        error = sub(delta, j) if valid else nan[:]
        values = {k: snapshot[k] for k in self.VECTOR_FIELDS if k in snapshot}
        values.update(delta_P=delta, dP_total_dt=scale(delta, 1/dt) if dt > 0 else nan[:],
                      J_external=self._impulses['external'],
                      J_wall=self._impulses['wall'], J_other=self._impulses['other'],
                      J_expected=j, impulse_balance_error=error,
                      momentum_balance_error=scale(error, 1/dt) if valid else nan[:])
        row = dict(step=step, time=time, dt=dt, solver_dt=solver_dt,
                   n_particles=len(snapshot['population']),
                   population_unchanged=int(self._population_ok),
                   every_step_sampled=int(self._sampling_ok),
                   force_budget_complete=int(complete), balance_valid=int(valid),
                   coupling_available=int(snapshot['coupling_available']),
                   F_coupling_residual_norm=norm(snapshot['F_coupling_residual']),
                   impulse_balance_error_norm=norm(error),
                   momentum_balance_error_norm=norm(values['momentum_balance_error']))
        for prefix, value in values.items():
            for i, axis in enumerate('xyz'):
                row[prefix + '_' + axis] = value[i]
        self._writer.writerow(row)
        self._file.flush()
        self._last_write_time = time
        self._last_write_p = snapshot['P_total'][:]
        self._clear_interval()

    def ResetBaseline(self):
        """Discard the pending interval, e.g. immediately after inlet shutdown."""
        if self._latest is None:
            raise RuntimeError('Call Execute() before resetting the baseline')
        self._last_write_time = self._latest[0]
        self._last_write_p = self._latest[3]['P_total'][:]
        self._clear_interval()

    def close(self):
        handle = getattr(self, '_file', None)
        if handle is not None and not handle.closed:
            if self._latest is not None and self._latest[0] > self._last_write_time:
                self._write()
            handle.close()

    def __del__(self):
        handle = getattr(self, '_file', None)
        if handle is not None and not handle.closed:
            handle.close()
