package contigo_orekit.orekitutils;

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

            var sunECEF  = transform.transformPosition(sunPV.getPosition());
            var moonECEF = transform.transformPosition(moonPV.getPosition());

            //Vector3D sp = sunECEF.getPosition();
            //Vector3D mp = moonECEF.getPosition();

            output[0][i][0] = sunECEF.getX();
            output[0][i][1] = sunECEF.getY();
            output[0][i][2] = sunECEF.getZ();

            output[1][i][0] = moonECEF.getX();
            output[1][i][1] = moonECEF.getY();
            output[1][i][2] = moonECEF.getZ();
        }

        return output;
    }
}