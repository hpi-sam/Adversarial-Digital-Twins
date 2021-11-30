package de.mdelab.morisia.comparch.simulator;

/**
 * Enumeration of all supported issue. For each issue type, there is exactly one
 * injector.
 * 
 * {@link #NO_CF2} refers to the injection of failures to a provided interface
 * while the number of failures is below the threshold to cause a CF2.
 * 
 * @author thomas vogel
 * @version 0.30
 *
 */
public enum IssueType {

	CF1("CF1"), CF2("CF2"), NO_CF2("NO_CF2"), CF3("CF3"), CF4("CF4"),CF5("CF5"),PI1("PI1"),PI2("PI2"),PI3("PI3");

	/**
	 * Issue type as a String.
	 */
	private String type;

	/**
	 * Constructor.
	 * 
	 * @param type
	 *            issue type as a String.
	 */
	private IssueType(String type) {
		this.type = type;
	}

	/**
	 * Returns the issue type as a String.
	 * 
	 * @return the issue type as a String.
	 */
	public String getType() {
		return this.type;
	}

	public static IssueType[] getAllValues() {
		return IssueType.class.getEnumConstants();
	}

}
