import torch


class LightweightKalmanFilter:
    """Stage-1 motion model.

    This simplified filter tracks center/size dynamics for OBB detections and keeps
    the rotation angle outside the state vector. Angle smoothing is handled in Track.
    """

    def __init__(self, dt=1.0):
        self.dt = float(dt)
        self.motion_mat = torch.eye(8, dtype=torch.float32)
        self.motion_mat[:4, 4:] = torch.eye(4, dtype=torch.float32) * self.dt
        self.update_mat = torch.eye(4, 8, dtype=torch.float32)
        self.std_weight_position = 1.0 / 20.0
        self.std_weight_velocity = 1.0 / 160.0

    def initiate(self, measurement):
        measurement = measurement.to(dtype=torch.float32)
        mean = torch.cat([measurement, torch.zeros(4, dtype=torch.float32)])
        std = self._state_std(measurement)
        covariance = torch.diag(std * std)
        return mean, covariance

    def predict(self, mean, covariance):
        mean = (self.motion_mat @ mean).to(dtype=torch.float32)
        std = self._process_std(mean[:4])
        motion_cov = torch.diag(std * std)
        covariance = self.motion_mat @ covariance @ self.motion_mat.t() + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        mean = (self.update_mat @ mean).to(dtype=torch.float32)
        std = self._measurement_std(mean)
        innovation_cov = torch.diag(std * std)
        covariance = self.update_mat @ covariance @ self.update_mat.t() + innovation_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        measurement = measurement.to(dtype=torch.float32)
        projected_mean, projected_cov = self.project(mean, covariance)
        cross_cov = covariance @ self.update_mat.t()
        kalman_gain = cross_cov @ torch.linalg.inv(projected_cov)
        innovation = measurement - projected_mean
        new_mean = mean + kalman_gain @ innovation
        new_covariance = covariance - kalman_gain @ projected_cov @ kalman_gain.t()
        return new_mean, new_covariance

    def _state_std(self, measurement):
        width = max(float(measurement[2].item()), 1.0)
        height = max(float(measurement[3].item()), 1.0)
        pos_xy = 2.0 * self.std_weight_position * max(width, height)
        pos_wh = 2.0 * self.std_weight_position * torch.tensor([width, height], dtype=torch.float32)
        vel_xy = 10.0 * self.std_weight_velocity * max(width, height)
        vel_wh = 10.0 * self.std_weight_velocity * torch.tensor([width, height], dtype=torch.float32)
        return torch.tensor([pos_xy, pos_xy, pos_wh[0], pos_wh[1], vel_xy, vel_xy, vel_wh[0], vel_wh[1]], dtype=torch.float32)

    def _process_std(self, measurement):
        width = max(float(measurement[2].item()), 1.0)
        height = max(float(measurement[3].item()), 1.0)
        pos_xy = self.std_weight_position * max(width, height)
        pos_wh = self.std_weight_position * torch.tensor([width, height], dtype=torch.float32)
        vel_xy = self.std_weight_velocity * max(width, height)
        vel_wh = self.std_weight_velocity * torch.tensor([width, height], dtype=torch.float32)
        return torch.tensor([pos_xy, pos_xy, pos_wh[0], pos_wh[1], vel_xy, vel_xy, vel_wh[0], vel_wh[1]], dtype=torch.float32)

    def _measurement_std(self, measurement):
        width = max(float(measurement[2].item()), 1.0)
        height = max(float(measurement[3].item()), 1.0)
        pos_xy = self.std_weight_position * max(width, height)
        pos_wh = self.std_weight_position * torch.tensor([width, height], dtype=torch.float32)
        return torch.tensor([pos_xy, pos_xy, pos_wh[0], pos_wh[1]], dtype=torch.float32)
