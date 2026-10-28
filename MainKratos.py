import sys
import time

import KratosMultiphysics
from KratosMultiphysics import Logger
from KratosMultiphysics.DEMApplication.DEM_analysis_stage import DEMAnalysisStage

from momentum_tracker import MomentumTracker
from particle_step_logger import ParticleStepLogger


class DEMAnalysisStageWithFlush(DEMAnalysisStage):

    def __init__(self, model, project_parameters, flush_frequency=10.0):
        super().__init__(model, project_parameters)

        self.flush_frequency = flush_frequency
        self.last_flush = time.time()

        # -------------------------------------------------------------
        # Suspended-particle force settings
        # -------------------------------------------------------------

        # Select the inlet-generated suspended particle(s), while excluding
        # smaller DPD fluid particles.
        self.minimum_target_radius = 0.19

        # The DEM inlet stops at t = 1.5 s. The driving force is applied at
        # the FIRST simulation time step after that time.
        self.force_start_time = 11.0

        # Prescribed external force on each selected suspended particle.
        # Example direction: +X.
        self.driving_force_x = 0.0
        self.driving_force_y = 0.0
        self.driving_force_z = 0.0

        # Parent model part containing DPD and suspended particles.
        self.particles_model_part_name = "SpheresPart"

        # Runtime state / diagnostics.
        self.target_nodes = []
        self.force_has_been_activated = False
        self.force_activation_time = None

        # -------------------------------------------------------------
        # Momentum tracking settings
        # -------------------------------------------------------------
        self.enable_momentum_tracker = True
        self.momentum_output_interval = 1.0e-1
        self.next_momentum_output_time = 0.0
        self.momentum_tracker = None

    def Initialize(self):
        super().Initialize()

        particles_model_part = self._GetParticlesModelPart()
        
        self.particle_step_logger = ParticleStepLogger(
            particles_model_part,
            start_time=5.6,
            end_time=6.0,
            filename="particle_step_debug.csv",
        )

        if self.enable_momentum_tracker:
                self.momentum_tracker = MomentumTracker(
                particles_model_part,
                output_directory=".",
                suspended_radius_threshold=self.minimum_target_radius,
                output_file_name="momentum_history.csv",
                output_interval=0.1,
                classification="flag",
                wall_mode="none",
                gravity=(0.0, 0.0, 0.0),
                forces_complete=True,
            )
                self.next_momentum_output_time = particles_model_part.ProcessInfo[
                    KratosMultiphysics.TIME]
                self.momentum_tracker.Execute()

        print("---- Suspended-particle force configuration ----")
        print(
            "Selection criterion: RADIUS >= {}".format(
                self.minimum_target_radius))
        print(
            "Force start rule: first step after t = {} s".format(
                self.force_start_time))
        print(
            "External force per selected particle: [{}, {}, {}]".format(
                self.driving_force_x,
                self.driving_force_y,
                self.driving_force_z))
        print("Particle model part: {}".format(
            self.particles_model_part_name))

    def _GetParticlesModelPart(self):
        if not self.model.HasModelPart(self.particles_model_part_name):
            raise RuntimeError(
                "Particle model part '{}' was not found. "
                "Available model parts: {}".format(
                    self.particles_model_part_name,
                    list(self.model.GetModelPartNames())))

        return self.model.GetModelPart(self.particles_model_part_name)

    def _GetTargetParticles(self):
        particles_model_part = self._GetParticlesModelPart()
        target_particles = []

        for node in particles_model_part.Nodes:
            radius = node.GetSolutionStepValue(KratosMultiphysics.RADIUS)

            if radius >= self.minimum_target_radius:
                target_particles.append(node)

        return target_particles

    def _CreateDrivingForce(self):
        force = KratosMultiphysics.Array3()
        force[0] = self.driving_force_x
        force[1] = self.driving_force_y
        force[2] = self.driving_force_z
        return force

    def _SetExternalForceOnTargetParticles(self, force):
        target_particles = self._GetTargetParticles()
        current_ids = set()

        for node in target_particles:
            node.SetSolutionStepValue(
                KratosMultiphysics.EXTERNAL_APPLIED_FORCE,
                force)
            current_ids.add(node.Id)

        self.target_particle_ids = current_ids
        return target_particles

    def InitializeSolutionStep(self):
        super().InitializeSolutionStep()
        
        self.particle_step_logger.BeginStep()

        if self.force_has_been_activated:
            return                      # nothing to do after the one-time assignment

        mp = self._GetParticlesModelPart()
        current_time = mp.ProcessInfo[KratosMultiphysics.TIME]

        if current_time < self.force_start_time:
            return

        self.target_nodes = self._GetTargetParticles()   # single scan
        if len(self.target_nodes) == 0:
            raise RuntimeError(
                "No suspended particle with RADIUS >= {} found at t = {}".format(
                    self.minimum_target_radius, current_time))

        force = self._CreateDrivingForce()
        for node in self.target_nodes:
            node.SetSolutionStepValue(
                KratosMultiphysics.EXTERNAL_APPLIED_FORCE, force)

        self.force_has_been_activated = True
        self.force_activation_time = current_time
        print("---- Driving force activated at t = {} on {} particle(s) ----".format(
            current_time, len(self.target_nodes)))

    def FinalizeSolutionStep(self):
        self.particle_step_logger.EndStep()
        super().FinalizeSolutionStep()

        if self.enable_momentum_tracker and self.momentum_tracker is not None:
            self.momentum_tracker.Execute()

        if self.parallel_type == "OpenMP":
            now = time.time()

            if now - self.last_flush > self.flush_frequency:
                sys.stdout.flush()
                self.last_flush = now

    def Finalize(self):
        self.particle_step_logger.close()
        
        if self.enable_momentum_tracker and self.momentum_tracker is not None:
            self.momentum_tracker.Execute()
            self.momentum_tracker.close()

        super().Finalize()


if __name__ == "__main__":
    Logger.GetDefaultOutput().SetSeverity(Logger.Severity.INFO)

    with open("ProjectParametersDEM.json", "r") as parameter_file:
        parameters = KratosMultiphysics.Parameters(
            parameter_file.read())

    global_model = KratosMultiphysics.Model()

    DEMAnalysisStageWithFlush(
        global_model,
        parameters).Run()