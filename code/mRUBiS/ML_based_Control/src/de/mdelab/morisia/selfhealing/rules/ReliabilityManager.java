package de.mdelab.morisia.selfhealing.rules;

import java.util.List;
import java.util.Random;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.ComponentType;

public class ReliabilityManager {

	private static Random random = new Random();

	private final static Logger LOGGER = Logger
			.getLogger(ReliabilityManager.class.getName());

	/**
	 * Updates the reliability of the given component type.
	 * 
	 * @param componentType
	 *            the given component type.
	 */
	private static void updateReliability(List<ComponentType> componentTypes) {
		double increase = 1 + (random.nextDouble() / 10);
		double decrease = 1 - (random.nextDouble() / 10);
		int selectedTypeForIncrease = random.nextInt(componentTypes.size());

		for (int i = 0; i < componentTypes.size(); i++) {
			ComponentType componentType = componentTypes.get(i);
			String componentTypeName = componentType.getName();
			double reliability = 0;
			if (componentTypeName.startsWith("Google")
					|| componentTypeName.startsWith("Twitter")
					|| componentTypeName.startsWith("Facebook")) {
				if (i == selectedTypeForIncrease) {
					reliability = componentType.getReliability() * increase;
					if (reliability > 1.0) {
						reliability = 1.0;
					}
				} else {
					reliability = componentType.getReliability() * decrease;
				}
			} else {
				// local components have always a reliability of 1.
				reliability = 1.0;
			}
			componentType.setReliability(reliability);
			LOGGER.info("Updated reliability of " + reliability
					+ " for the component type " + componentType);

		}
	}

}
