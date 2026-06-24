import jax.numpy as jnp
from hj_reachability import dynamics
from hj_reachability import sets


class DoubleIntegrator(dynamics.ControlAndDisturbanceAffineDynamics):
    """
    Double Integrator affine dynamical system class.
    
    Dynamics:
        x_dot = v
        v_dot = u + d
    
    State:
        [position, velocity]
    """

    def __init__(self,
                 u_max=3.0,
                 d_max=0.0,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):
        
        # Set default bounding boxes for control [-u_max, u_max]
        if control_space is None:
            control_space = sets.Box(jnp.array([-u_max]), jnp.array([u_max]))
            
        # Set default bounding boxes for disturbance [-d_max, d_max]
        if disturbance_space is None:
            disturbance_space = sets.Box(jnp.array([-d_max]), jnp.array([d_max]))
            
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        """
        The baseline system drift f(x, t) without control or disturbance inputs.
        x_dot = v
        v_dot = 0
        """
        _, v = state
        return jnp.array([v, 0.0])

    def control_jacobian(self, state, time):
        """
        The control multiplier matrix B(x, t).
        Control 'u' directly affects only the velocity derivative (index 1).
        """
        return jnp.array([
            [0.0],  # u does not directly affect position change
            [1.0],  # u directly adds acceleration to velocity
        ])

    def disturbance_jacobian(self, state, time):
        """
        The disturbance multiplier matrix C(x, t).
        Disturbance 'd' directly affects only the velocity derivative (index 1).
        """
        return jnp.array([
            [0.0],  # d does not directly affect position change
            [1.0],  # d acts alongside control on acceleration
        ])