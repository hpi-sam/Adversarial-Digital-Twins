package de.mdelab.morisia.comparch.simulator;

import de.mdelab.morisia.comparch.Architecture;

/**
 * Analyzer of the model.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public interface Analyzer {

	/**
	 * @return the name of the analyzer.
	 */
	public String getName();

	/**
	 * @return the description of the analyzer.
	 */
	public String getDescription();

	/**
	 * Analyzes the model and returns the number of failures identified by the
	 * analysis in the model.
	 * 
	 * @param eArchitecture
	 *            the root element of the model
	 * @return the number of failures identified in the model.
	 */
	public int analyze(Architecture eArchitecture);

}
