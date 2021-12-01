package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model;

import java.util.List;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Failure;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.ModelAnalyzer;

/**
 * Analyzes if and how many {@link Failure} elements are present in the model.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class AttachedFailureAnalyzer implements ModelAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(AttachedFailureAnalyzer.class.getName());

	private int failureThreshold;

	/**
	 * Constructor.
	 * 
	 * @param failureThreshold
	 *            threshold for number of failures in the model.
	 */
	public AttachedFailureAnalyzer(int failureThreshold) {
		this.failureThreshold = failureThreshold;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Attached Failure Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes if and how many Failure elements are present in the model. "
				+ "For one provided interface, the number of failures should be below "
				+ this.failureThreshold;
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

			int currentNumberOfFailures = failureCount;

			for (Component eComponent : eTenant.getComponents()) {
				LOGGER.finer("...... Component: \"" + eComponent.getUid()
						+ "\"");
				for (ProvidedInterface eProInterface : eComponent
						.getProvidedInterfaces()) {
					List<Failure> eFailures = eProInterface.getFailures();
					if (eFailures == null || eFailures.size() == 0) {
						LOGGER.finer("[OK] Provided interface \""
								+ eProInterface.getUid()
								+ "\" has no failures.");
					} else {
						if (eFailures.size() >= this.failureThreshold) {
							LOGGER.severe("[FAILURE] Provided interface \""
									+ eProInterface.getUid() + "\" has "
									+ eFailures.size() + " failures.");
							failureCount++;
						} else {
							LOGGER.warning("Provided interface \""
									+ eProInterface.getUid() + "\" has "
									+ eFailures.size() + " failures.");
						}

					}
				}
			}

			int numberOfFailuresInTenant = failureCount
					- currentNumberOfFailures;
			if (numberOfFailuresInTenant == 0) {
				LOGGER.info("[OK] No provided interface has attached too many failures.\n");
			} else {
				LOGGER.severe("[FAILURES] "
						+ numberOfFailuresInTenant
						+ " provided interfaces have attached too many failures.\n");
			}

		} // next tenant
		return failureCount;
	}

}
