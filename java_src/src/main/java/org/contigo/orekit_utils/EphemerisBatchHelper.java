package org.contigo.orekit_utils;

import org.orekit.frames.*;
import org.orekit.utils.IERSConventions;
import org.orekit.bodies.CelestialBody;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.time.AbsoluteDate;
import java.util.List;
import java.util.ArrayList;

public class EphemerisBatchHelper {

    public static double[][][] getBodyECEF(
            AbsoluteDate refDate,
            double[] offsetsSeconds,
            List<String> bodyNames) {

        int numBodies = bodyNames.size();
        int numSteps = offsetsSeconds.length;
        double[][][] output = new double[numBodies][numSteps][3];

        // 1. Resolve body objects once and store them
        List<CelestialBody> bodies = new ArrayList<>(numBodies);
        for (String name : bodyNames) {
            bodies.add(CelestialBodyFactory.getBody(name));
        }

        Frame eci  = FramesFactory.getEME2000();
        Frame ecef = FramesFactory.getITRF(IERSConventions.IERS_2010, true);

        // 2. Outer loop for time (Calculates the ECI -> ECEF matrix once per step)
        for (int t = 0; t < numSteps; t++) {
            AbsoluteDate date = refDate.shiftedBy(offsetsSeconds[t]);
            StaticTransform transform = eci.getStaticTransformTo(ecef, date);

            // 3. Inner loop for bodies
            for (int b = 0; b < numBodies; b++) {
                var posInEci = bodies.get(b).getPosition(date, eci);
                var posInEcef = transform.transformPosition(posInEci);

                output[b][t][0] = posInEcef.getX();
                output[b][t][1] = posInEcef.getY();
                output[b][t][2] = posInEcef.getZ();
            }
        }

        return output;
    }
}