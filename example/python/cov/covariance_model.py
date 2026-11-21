# Build a covariance model with the cov helper, look at its text and its
# g2s arguments, and evaluate it on the server (covEval) along two directions.
import numpy as np
import matplotlib.pyplot as plt
from g2s import g2s, cov

# 10% nugget and a spherical structure: practical range 40 cells along the
# major axis, 15 along the minor axis, major axis turned by 30 degrees
model = 0.1*cov.nugget() + 0.9*cov.spherical(range=[40, 15], angle=30)
print(str(model))    # 0.1*nugget + 0.9*spherical(range=[40, 15], angle=30)
print(model.args())  # ['-cov', '0.1*nugget + 0.9*spherical(range=[40, 15], angle=30)']

# Lags are [column shift, row shift] (component 0 = last numpy axis)
distance = np.arange(0, 51)
theta = np.radians(30)
major = np.column_stack([distance*np.cos(theta), distance*np.sin(theta)])
minor = np.column_stack([-distance*np.sin(theta), distance*np.cos(theta)])
gamma_major = model.variogram(major)  # runs covEval on the G2S server
gamma_minor = model.variogram(minor)

# The same evaluation without the helper
direct = g2s('-a', 'covEval', '-cov', str(model), '-lag', major.astype(np.float32), '-vario')
assert np.allclose(direct['variogram'], gamma_major)

plt.plot(distance, gamma_major, label='major axis (30 degrees)')
plt.plot(distance, gamma_minor, label='minor axis (120 degrees)')
plt.xlabel('lag (cells)')
plt.ylabel('variogram')
plt.legend()
plt.title(str(model), fontsize=8)
plt.show()
