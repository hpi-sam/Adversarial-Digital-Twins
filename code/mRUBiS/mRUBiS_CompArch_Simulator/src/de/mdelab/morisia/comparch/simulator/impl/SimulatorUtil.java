package de.mdelab.morisia.comparch.simulator.impl;

import java.util.Random;

import org.eclipse.emf.ecore.util.EcoreUtil;

/**
 * Utility class of the simulator.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class SimulatorUtil {

	// Usually this should be a field rather than a method variable so
	// that it is not re-seeded every call.
	private static Random rand = new Random();

	/**
	 * Returns a pseudo-random number between min and max, inclusive. The
	 * difference between min and max can be at most
	 * <code>Integer.MAX_VALUE - 1</code>.
	 * 
	 * @param min
	 *            Minimum value
	 * @param max
	 *            Maximum value. Must be greater than min.
	 * @return Integer between min and max, inclusive.
	 * @see java.util.Random#nextInt(int)
	 */
	public static int randInt(int min, int max) {

		// nextInt is normally exclusive of the top value,
		// so add 1 to make it inclusive
		int randomNum = rand.nextInt((max - min) + 1) + min;

		return randomNum;
	}

	/**
	 * Returns a generated UUID.
	 * 
	 * @return a generated UUID.
	 */
	public static String generateUUID() {
		return EcoreUtil.generateUUID();
	}

	public final static String SEPARATOR = "\n----------------------------------------------------------------------------------------------\n";
	public final static String DOUBLE_SEPARATOR = "\n==============================================================================================\n";

}
