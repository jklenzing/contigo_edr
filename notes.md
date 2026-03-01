


```java
# ================================================================
# JAVA BATCH HELPER APPROACH (Removes Python Loop Entirely)
# ================================================================
#
# Strategy:
# 1. Move entire Sun/Moon + transform loop into Java
# 2. Pass time offsets as double[]
# 3. Return double[][][] of shape [2][N][3]
# 4. Convert once to NumPy
#
# This eliminates thousands of Python↔JVM crossings.
# Expected speedup for large N (100k+): 10–20×
# ================================================================

# ================================================================
# STEP 1 — Java Helper Class (compile into jar)
# ================================================================

# File: fast/EphemerisBatchHelper.java

package fast;

import org.orekit.frames.*;
import org.orekit.time.*;
import org.orekit.utils.*;
import org.orekit.bodies.*;
import org.orekit.utils.IERSConventions;
import org.hipparchus.geometry.euclidean.threed.Vector3D;

public class EphemerisBatchHelper {

    public static double[][][] getSunMoonECEF(
            AbsoluteDate refDate,
            double[] offsetsSeconds) {

        int N = offsetsSeconds.length;
        double[][][] output = new double[2][N][3];

        Frame eci  = FramesFactory.getEME2000();
        Frame ecef = FramesFactory.getITRF(IERSConventions.IERS_2010, true);

        CelestialBody sun  = CelestialBodyFactory.getSun();
        CelestialBody moon = CelestialBodyFactory.getMoon();

        for (int i = 0; i < N; i++) {

            AbsoluteDate date = refDate.shiftedBy(offsetsSeconds[i]);

            var sunPV  = sun.getPVCoordinates(date, eci);
            var moonPV = moon.getPVCoordinates(date, eci);

            var transform = eci.getStaticTransformTo(ecef, date);

            var sunECEF  = transform.transformPVCoordinates(sunPV);
            var moonECEF = transform.transformPVCoordinates(moonPV);

            Vector3D sp = sunECEF.getPosition();
            Vector3D mp = moonECEF.getPosition();

            output[0][i][0] = sp.getX();
            output[0][i][1] = sp.getY();
            output[0][i][2] = sp.getZ();

            output[1][i][0] = mp.getX();
            output[1][i][1] = mp.getY();
            output[1][i][2] = mp.getZ();
        }

        return output;
    }
}

```

```python
# Compile and package:
# javac -cp orekit.jar:hipparchus-core.jar fast/EphemerisBatchHelper.java
# jar cf fasthelper.jar fast/EphemerisBatchHelper.class


# ================================================================
# STEP 2 — Python Wrapper
# ================================================================

from org.orekit.time import AbsoluteDate, TimeScalesFactory
import numpy as np

# IMPORTANT:
# Start JVM with your jar BEFORE importing Orekit
# orekit.initVM(classpath=['fasthelper.jar'])

from fast import EphemerisBatchHelper


def get_sun_moon_ecef(times):

    utc = TimeScalesFactory.getUTC()

    first_dt = times[0]
    ref_date = AbsoluteDate(
        first_dt.year,
        first_dt.month,
        first_dt.day,
        first_dt.hour,
        first_dt.minute,
        float(first_dt.second),
        utc
    )

    offsets = np.array(
        [(t - first_dt).total_seconds() for t in times],
        dtype=np.float64
    )

    # Single JVM call (entire batch handled in Java)
    result = EphemerisBatchHelper.getSunMoonECEF(ref_date, offsets)

    # Convert to NumPy once
    return np.array(result, dtype=np.float64)


# ================================================================
# PERFORMANCE CHARACTERISTICS
# ================================================================
# Python Loop Version:
#   ~6–10 JVM crossings per epoch
#
# Java Batch Version:
#   1 JVM crossing total
#
# For N = 100,000 epochs:
#   Python version: seconds to minutes
#   Java batch: typically 10–20× faster
#
# If needed, we can push further by:
# - Reusing frames outside method (static initialization)
# - Using preallocated static arrays
# - Removing transform inside loop if frame fixed
# ================================================================
```

```
contigo_edr/
│
├── contigo_edr/
│   ├── __init__.py
│   ├── java/
│   │   └── fasthelper.jar
│   ├── ephemeris.py
│   └── ...
│
├── pyproject.toml
└── setup.cfg
```

JVM Heap Size

By default, Python-started JVM may only get ~512MB–1GB heap.

If you plan to process millions of points:

You should explicitly set heap size:

```python
orekit.initVM(
    classpath=[jar_path],
    vmargs=['-Xms512m', '-Xmx4g']
)
```