package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model;

import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.Parameter;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.ModelAnalyzer;

/**
 * Analyzes whether each {@link Parameter} in the model has a value assigned.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class ParameterAnalyzer implements ModelAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(ParameterAnalyzer.class.getName());

	/**
	 * Constructor.
	 */
	public ParameterAnalyzer() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Parameter Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes whether each Parameter in the model has a value assigned.";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int analyze(Architecture eArchitecture) {
		int failureCount = 0;

		for (Tenant eTenant : eArchitecture.getTenants()) {
			LOGGER.info("Analyzing the tenant \"" + eTenant.getName()
					+ "\" ...");

			boolean propertiesFound = false;
			int currentNumberOfFailures = failureCount;

			for (Component eComponent : eTenant.getComponents()) {
				if (eComponent.getState() == ComponentLifeCycle.STARTED) {
					for (Parameter eParameter : eComponent.getParameters()) {
						propertiesFound = true;
						Object value = eParameter.getValue();
						if (value == null) {
							LOGGER.severe("[FAILURE] The parameter \""
									+ eParameter.getType().getName()
									+ "\" of component \""
									+ eComponent.getUid()
									+ "\" has no value assigned.");
							failureCount++;
						} else {
							LOGGER.finer("[OK] The parameter \""
									+ eParameter.getType().getName()
									+ "\" of component \""
									+ eComponent.getUid()
									+ "\" has assigned a value of " + value);
						}
					}

				}
			}

			if (!propertiesFound) {
				LOGGER.info("[OK] No parameters have been found.");
			}

			// log summary
			int numberOfFailuresInTenant = failureCount
					- currentNumberOfFailures;
			if (numberOfFailuresInTenant == 0) {
				LOGGER.info("[OK] All parameters have a value assigned.\n");
			} else {
				LOGGER.severe("[FAILURES] " + numberOfFailuresInTenant
						+ " parameters have no value assigned.");
			}

		} // next tenant

		return failureCount;
	}

}
