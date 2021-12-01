package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.adaptation;

import java.util.Queue;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.Connector;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.RequiredInterface;
import de.mdelab.morisia.comparch.simulator.AdaptationAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisNames;
import de.mdelab.morisia.comparch.simulator.mrubis.events.ChangeEventQueue;
import de.mdelab.morisia.comparch.simulator.mrubis.events.ReroutingEvent;

/**
 * Analyzes an adaptation performed by a feedback loop on the model. The
 * adaptation replaces a component in the model with an alternative component.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class ComponentReplacementAnalyzer implements AdaptationAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(ComponentReplacementAnalyzer.class.getName());

	/**
	 * Constructor.
	 */
	public ComponentReplacementAnalyzer() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Component Replacement Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes a component replacement adaptation based on re-routings.";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int analyze(Architecture eArchitecture) {

		int failureCount = 0;

		Queue<ReroutingEvent> events = ChangeEventQueue.INSTANCE
				.getReroutingEvents();

		LOGGER.info("\n----------------------------------------------------------------------------------------------\n"
				+ "\tAadaptation Analyzer: "
				+ events.size()
				+ " rerouting events to be checked ..."
				+ "\n----------------------------------------------------------------------------------------------");

		while (events.peek() != null) {
			ReroutingEvent event = events.poll();
			RequiredInterface requiredInterface = event.getRequiredInterface();
			Component compOfRi = requiredInterface.getComponent();
			String tenantName = null;
			if (compOfRi.getTenant() != null) {
				tenantName = compOfRi.getTenant().getName();
			}

			// BEGIN LOG FINER
			String msg = "Reroute required interface \""
					+ requiredInterface.getUid() + "\" of component \""
					+ compOfRi.getUid() + "\" of tenant \"" + tenantName + "\"";
			Connector o = event.getOldConnector();
			if (o != null) {
				ProvidedInterface pi = o.getTarget();
				if (pi != null) {
					Component oldTarget = pi.getComponent();
					msg = msg + "\n\tfrom \"" + oldTarget.getUid() + "\"";
				} else {
					msg = msg + "\n\tfrom null";
				}
			} else {
				msg = msg + "\n\tfrom null";
			}

			Connector n = event.getNewConnector();
			if (n != null) {
				ProvidedInterface pi = n.getTarget();
				if (pi != null) {
					Component newTarget = pi.getComponent();
					msg = msg + "\n\tto \"" + newTarget.getUid() + "\"";
				} else {
					msg = msg + "\n\tto null";
				}

			} else {
				msg = msg + "\n\tto null";
			}
			LOGGER.finer(msg);
			// END LOG FINER DEBUG

			if (requiredInterface.getType().getFullyQualifiedName()
					.equals(MrubisNames.AUTHENTICATION_SERVICE_INTERFACE)) {
				Connector oldConnector = event.getOldConnector();
				Connector newConnector = event.getNewConnector();
				if (oldConnector == null) {
					if (newConnector == null) {
						LOGGER.warning("Required interface \""
								+ requiredInterface.getUid()
								+ "\" of component \"" + compOfRi.getUid()
								+ "\" of tenant \"" + tenantName
								+ "\" is still disconnected (no connector).");
						failureCount++;

					} else {
						ProvidedInterface providedInterface = newConnector
								.getTarget();
						if (providedInterface != null) {
							Component providedComponent = providedInterface
									.getComponent();
							ComponentType providedComponentType = providedComponent
									.getType();
							LOGGER.info("Required interface \""
									+ requiredInterface.getUid()
									+ "\" of component \""
									+ compOfRi.getUid()
									+ "\" of tenant \""
									+ tenantName
									+ "\" is now connected to an interface provided by component \""
									+ providedComponent.getUid()
									+ "\" of type \""
									+ providedComponentType.getName() + "\".");

						} else {
							LOGGER.warning("Required interface \""
									+ requiredInterface.getUid()
									+ "\" of component \""
									+ compOfRi.getUid()
									+ "\" of tenant \""
									+ tenantName
									+ "\" is still disconnected (connector with target = null).");
							failureCount++;
						}
					}
				} else {
					// oldConnector != null
					if (newConnector == null) {
						LOGGER.warning("Required interface \""
								+ requiredInterface.getUid()
								+ "\" of component \"" + compOfRi.getUid()
								+ "\" of tenant \"" + tenantName
								+ "\" is now disconnected (no connector).");
						failureCount++;
					} else {
						ProvidedInterface providedInterface = newConnector
								.getTarget();
						if (providedInterface != null) {
							Component providedComponent = providedInterface
									.getComponent();
							ComponentType providedComponentType = providedComponent
									.getType();

							LOGGER.info("Required interface \""
									+ requiredInterface.getUid()
									+ "\" of component \""
									+ compOfRi.getUid()
									+ "\" of tenant \""
									+ tenantName
									+ "\" is now connected to an interface provided by component \""
									+ providedComponent.getUid()
									+ "\" of type \""
									+ providedComponentType.getName() + "\".");

							ProvidedInterface oldProvidedInterface = oldConnector
									.getTarget();

							if (oldProvidedInterface != null) {
								Component oldProvidedComponent = oldProvidedInterface
										.getComponent();
								ComponentType oldProvidedComponentType = oldProvidedComponent
										.getType();
								LOGGER.info(" -- Before it was connected to an interface provided by component \""
										+ oldProvidedComponent.getUid()
										+ "\" of type \""
										+ oldProvidedComponentType.getName()
										+ "\".");
							}

						} else {

							LOGGER.warning("Required interface \""
									+ requiredInterface.getUid()
									+ "\" of component \""
									+ compOfRi.getUid()
									+ "\" of tenant \""
									+ tenantName
									+ "\" is still disconnected (connector with target = null).");
							failureCount++;
						}
					}
				}
			}

		}

		return failureCount;
	}

}
