package org.contigo.orekit_utils;

import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;
import org.orekit.orbits.CartesianOrbit;
import org.orekit.propagation.SpacecraftState;
import org.orekit.forces.radiation.SolarRadiationPressure;
import org.orekit.forces.radiation.IsotropicRadiationSingleCoefficient;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.bodies.CelestialBody;
import org.orekit.bodies.OneAxisEllipsoid;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.orekit.frames.Transform;

public class SRPCannonballBatchHelper {

    private static final Frame ECEF =
            FramesFactory.getITRF(IERSConventions.IERS_2010, true);

    private static final Frame ECI =
            FramesFactory.getEME2000();

    private static final CelestialBody SUN =
            CelestialBodyFactory.getSun();

    private static final OneAxisEllipsoid EARTH =
            new OneAxisEllipsoid(
                    Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
                    Constants.WGS84_EARTH_FLATTENING,
                    ECEF);

    /**
     * Batch SRP acceleration computation.
     *
     * @param referenceDate  reference AbsoluteDate
     * @param offsetSeconds  array (N) of time offsets from reference date (seconds)
     * @param states         array (N,6) spacecraft states in ECEF [x,y,z,vx,vy,vz]
     * @param mass           spacecraft mass (kg)
     * @param area           spacecraft SRP area (m^2)
     * @param cr             coefficient of reflection
     * @return double[N][6] -> {ax_eci, ay_eci, az_eci, ax_ecef, ay_ecef, az_ecef}
     */
    public static double[][] get_acc(
            AbsoluteDate referenceDate,
            double[] offsetSeconds,
            double[][] states,
            double mass,
            double area,
            double cr) {

        int n = offsetSeconds.length;

        if (states.length != n) {
            throw new IllegalArgumentException("offsetSeconds and states must have same length");
        }

        double[][] output = new double[n][6];

        // Surface and SRP model can be reused across loop
        IsotropicRadiationSingleCoefficient surface =
                new IsotropicRadiationSingleCoefficient(area, cr);

        SolarRadiationPressure srp =
                new SolarRadiationPressure(SUN, EARTH, surface);

        for (int i = 0; i < n; i++) {

            AbsoluteDate date = referenceDate.shiftedBy(offsetSeconds[i]);

            double[] state = states[i];

            Vector3D positionEcef = new Vector3D(state[0], state[1], state[2]);
            Vector3D velocityEcef = new Vector3D(state[3], state[4], state[5]);

            PVCoordinates pvEcef = new PVCoordinates(positionEcef, velocityEcef);

            // Transform to ECI
            Transform transform = ECEF.getTransformTo(ECI, date);
            PVCoordinates pvEci = transform.transformPVCoordinates(pvEcef);

            CartesianOrbit orbit = new CartesianOrbit(
                    pvEci,
                    ECI,
                    date,
                    Constants.WGS84_EARTH_MU);

            SpacecraftState scState = new SpacecraftState(orbit).withMass(mass);

            Vector3D accelEci = srp.acceleration(scState, srp.getParameters());

            Vector3D accelEcef =
                    ECI.getStaticTransformTo(ECEF, date)
                       .getRotation()
                       .applyTo(accelEci);

            output[i][0] = accelEci.getX();
            output[i][1] = accelEci.getY();
            output[i][2] = accelEci.getZ();
            output[i][3] = accelEcef.getX();
            output[i][4] = accelEcef.getY();
            output[i][5] = accelEcef.getZ();
        }

        return output;
    }
}
