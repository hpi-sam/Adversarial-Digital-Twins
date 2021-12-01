package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model;

import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.Connector;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.RequiredInterface;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.ModelAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisNames;

/**
 * Analyzes the {@link RequiredInterface}s in the model and checks whether they
 * are properly connected to provided interfaces of started components.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class RequiredInterfaceAnalyzer implements ModelAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(RequiredInterfaceAnalyzer.class.getName());

	/**
	 * Constructor.
	 */
	public RequiredInterfaceAnalyzer() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Required Interface Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes whether the required interfaces of started components "
				+ "are properly connected to other started components.";
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

			// last item filters are started filters that have no subsequent
			// filter. For each tenant, there should be at most 1.
			int numberOfLastItemFilters = 0;
			int currentNumberOfFailures = failureCount;

			for (Component eComponent : eTenant.getComponents()) {
				if (eComponent.getState() == ComponentLifeCycle.STARTED) {
					LOGGER.finer("......Started Component: \""
							+ eComponent.getUid() + "\"");
					if (eComponent.getRequiredInterfaces().size() == 0) {
						LOGGER.finer("[OK] No required interfaces.");
					} else {
						for (RequiredInterface eReqInterface : eComponent
								.getRequiredInterfaces()) {
							Connector eConnector = eReqInterface.getConnector();
							if (eConnector == null) {
								// there is no connector
								if (eReqInterface
										.getType()
										.getFullyQualifiedName()
										.equals(MrubisNames.ITEM_FILTER_SERVICE_INTERFACE)) {
									// filter components
									LOGGER.warning("The required interface \""
											+ eReqInterface.getUid()
											+ "\" has no connector.");
									numberOfLastItemFilters++;
								} else {
									// other components
									LOGGER.severe("[FAILURE] The required interface \""
											+ eReqInterface.getUid()
											+ "\" has no connector.");
									failureCount++;
								}
							} else {
								// there is a connector
								ProvidedInterface eProInterface = eConnector
										.getTarget();
								if (eProInterface == null) {
									// connector points to null
									LOGGER.severe("[FAILURE] The required interface \""
											+ eReqInterface.getUid()
											+ "\" has a connector that does not point to a provided interface (Target is null).");
									failureCount++;
								} else {
									// connector points to a provided interface
									if (!eProInterface.getType().equals(
											eReqInterface.getType())) {
										LOGGER.severe("[FAILURE] The required interface \""
												+ eReqInterface.getUid()
												+ "\" has a connector that points to a non-matching interface.\n"
												+ "          - Source Type: \""
												+ eReqInterface
														.getType()
														.getFullyQualifiedName()
												+ "\"\n"
												+ "          - Target: \""
												+ eProInterface.getUid()
												+ "\"\n"
												+ "          - Target Type: \""
												+ eProInterface
														.getType()
														.getFullyQualifiedName()
												+ "\"");
										failureCount++;
									} else {
										// Interface types match
										Component eTargetComponent = eProInterface
												.getComponent();
										if (eTargetComponent.getState() != ComponentLifeCycle.STARTED) {
											LOGGER.severe("[FAILURE] The required interface \""
													+ eReqInterface.getUid()
													+ "\" has a connector that properly points to an interface "
													+ "but the component \""
													+ eTargetComponent.getUid()
													+ "\" providing this interface is not STARTED, but "
													+ eTargetComponent
															.getState());
											failureCount++;
										} else {
											// requiring and providing
											// components
											// should belong to the same tenant
											if (!eTenant
													.equals(eTargetComponent
															.getTenant())) {
												LOGGER.severe("[FAILURE] The required interface \""
														+ eReqInterface
																.getUid()
														+ "\" of component \""
														+ eComponent.getUid()
														+ "\" from the tenant \""
														+ eTenant.getName()
														+ "\" is connected to the component \""
														+ eTargetComponent
																.getUid()
														+ "\" from the different tenant \""
														+ eTargetComponent
																.getTenant()
																.getName()
														+ "\".\n          Connected components must belong to the same tenant!");
												failureCount++;
											} else {
												LOGGER.finer("[OK] The required interface \""
														+ eReqInterface
																.getUid()
														+ "\" is properly connected to \""
														+ eProInterface
																.getUid()
														+ "\".");
											}
										}
									}
								}
							}

						}
					}

				}
			}
			if (numberOfLastItemFilters > 1) {
				LOGGER.severe("[FAILURE] More than one STARTED filter component, namely, "
						+ numberOfLastItemFilters
						+ ", "
						+ "have no connector to a subsequent filter component.");
				// increase failure count by the number of filters that have
				// no subsequent filter - 1. Minus 1 since the last filter
				// of the pipe has no subsequent filter, which is no
				// failure.
				failureCount = failureCount + numberOfLastItemFilters - 1;
			} else if (numberOfLastItemFilters == 1) {
				LOGGER.info("[OK] Exactly one STARTED filter "
						+ "has no connector to a subsequent filter component. "
						+ "This filter component ought to be the last filter of this tenant's pipe.");
			}

			// log summary
			int numberOfFailuresInTenant = failureCount
					- currentNumberOfFailures;
			if (numberOfFailuresInTenant == 0) {
				LOGGER.info("[OK] All required interfaces are properly connected.\n");
			} else {
				LOGGER.severe("[FAILURES] " + numberOfFailuresInTenant
						+ " required interfaces are not properly connected.\n");
			}

		} // next tenant
		return failureCount;
	}

}
