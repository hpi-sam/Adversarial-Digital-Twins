package de.mdelab.morisia.comparch.simulator;

/**
 * The types of self-adaptation capabilities supported by the simulator.
 * 
 * <ul>
 * <li>{@link Capability#SELF_REPAIR} to support the injection of CFs and the
 * corresponding analysis of the model.</li>
 * <li>{@link Capability#SELF_OPTIMIZATION} to support the injection of PIs and
 * the corresponding analysis of the model.</li>
 * <li>{@link Capability#SELF_MANAGEMENT} to support the injection of CFs and
 * PIs and the corresponding analysis of the model.</li>
 * <li>{@link Capability#BLANK} to obtain a blank simulator whose injectors and
 * analyzers must be manually added to the simulator instance before starting
 * the simulation.</li>
 * </ul>
 * 
 * @author thomas vogel
 * @version 0.30
 *
 */
public enum Capability {

	SELF_REPAIR("Self-Repair"), SELF_OPTIMIZATION("Self-Optimization"), SELF_MANAGEMENT(
			"Self-Management"), BLANK("Blank");

	/**
	 * Capability type as a String.
	 */
	private String type;

	/**
	 * Constructor.
	 * 
	 * @param type
	 *            Capability type as a String.
	 */
	private Capability(String type) {
		this.type = type;
	}

	/**
	 * @return the Capability type as a String.
	 */
	public String getType() {
		return this.type;
	}

}
