package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.adaptation;

import java.util.HashSet;
import java.util.Queue;
import java.util.Set;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.AdaptationAnalyzer;
import de.mdelab.morisia.comparch.simulator.impl.SimulatorUtil;
import de.mdelab.morisia.comparch.simulator.mrubis.events.ChangeEventQueue;
import de.mdelab.morisia.comparch.simulator.mrubis.events.LifecycleChangeEvent;

/**
 * Analyzes adaptations such as restarts or re-deployments based on the changes
 * of life cycle states of components in the model.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class LifecycleAdaptationAnalyzer implements AdaptationAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(LifecycleAdaptationAnalyzer.class.getName());

	/**
	 * Constructor.
	 */
	public LifecycleAdaptationAnalyzer() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Lifecycle Adaptation Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes adaptations such as restarts or re-deployments based on "
				+ "the changes of life cycle states of components in the model.";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int analyze(Architecture eArchitecture) {

		Set<Component> componentsToSetAsNotSupported = new HashSet<Component>();
		Queue<LifecycleChangeEvent> events = ChangeEventQueue.INSTANCE
				.getLifecycleEvents();

		LOGGER.info("\t" + this.getName() + ": " + events.size()
				+ " life cycle events to be checked ...");

		if (events.size() == 0) {
			LOGGER.info("\tNothing to analyze.");
		} else {
			while (events.peek() != null) {
				LifecycleChangeEvent event = events.poll();
				String componentId = event.getComponentId();
				Component eComponent = this.findComponentById(eArchitecture,
						componentId);
				if (eComponent != null) {
					ComponentLifeCycle source = event.getOldState();
					ComponentLifeCycle target = event.getNewState();

					Tenant eTenant = eComponent.getTenant();
					if (eTenant != null) {
						LOGGER.info("Component \"" + eComponent.getUid()
								+ "\" of tenant \"" + eTenant.getName() + "\"");
					} else {
						LOGGER.info("Component \"" + eComponent.getUid()
								+ "\" of tenant null");
					}

					boolean valid = this.isValidTransition(source, target);
					if (!valid) {
						componentsToSetAsNotSupported.add(eComponent);
					} else {
						// erase failures
						if (this.shouldFailuresBeErased(source, target)) {

							// clear failure elements attached to provided
							// interfaces of the component when the transitions
							// stop or undeploy' are taken.
							for (ProvidedInterface providedInterface : eComponent
									.getProvidedInterfaces()) {
								providedInterface.getFailures().clear();
							}

							LOGGER.info("Potential failures occurred when using a "
									+ "provided interface of component \""
									+ eComponent.getUid()
									+ "\" have been addressed.");
						}
					}
				} else {
					LOGGER.warning("Component \"" + event.getComponentId()
							+ "\" does not exist any more.");
				}
			}

			for (Component c : componentsToSetAsNotSupported) {
				c.setState(ComponentLifeCycle.NOT_SUPPORTED);

				String tenantName = null;
				if (c.getTenant() != null) {
					tenantName = c.getTenant().getName();
				}

				LOGGER.warning("INVALID LIFECYCLE OPERATION: Component \""
						+ c.getUid() + "\" of tenant \"" + tenantName
						+ "\" is set to NOT_SUPPORTED.");

			}

			LOGGER.info(SimulatorUtil.SEPARATOR + "\n");

		}

		return 0;
	}

	private Component findComponentById(Architecture eArchitecture,
			String componentId) {
		Component result = null;
		outerloop: for (Tenant eTenant : eArchitecture.getTenants()) {
			for (Component component : eTenant.getComponents()) {
				if (component.getUid().equals(componentId)) {
					result = component;
					break outerloop;
				}
			}
		}
		return result;
	}

	/**
	 * Checks whether a life cycle state change is valid.
	 * 
	 * @param source
	 *            the source state
	 * @param target
	 *            the target state
	 * @return <code>true</code> if it is valid, otherwise <code>false</code>.
	 */
	private boolean isValidTransition(ComponentLifeCycle source,
			ComponentLifeCycle target) {
		boolean isValid = false;

		String msg = "   -- Transition check: " + source + " => " + target
				+ " = ";

		// deploy
		if (source == ComponentLifeCycle.UNDEPLOYED
				& target == ComponentLifeCycle.DEPLOYED) {
			isValid = true;
			msg += "------------ DEPLOY ------------";

		} else

		// undeploy
		if (source == ComponentLifeCycle.DEPLOYED
				& target == ComponentLifeCycle.UNDEPLOYED) {
			isValid = true;
			msg += "------------ UNDEPLOY ------------";
		} else if (source == ComponentLifeCycle.NOT_SUPPORTED
				& target == ComponentLifeCycle.UNDEPLOYED) {
			isValid = true;
			msg += "------------ UNDEPLOY' ------------";
		} else

		// start
		if (source == ComponentLifeCycle.DEPLOYED
				& target == ComponentLifeCycle.STARTED) {
			isValid = true;
			msg += "------------ START ------------";
		} else

		// stop
		if ((source == ComponentLifeCycle.STARTED || source == ComponentLifeCycle.NOT_SUPPORTED)
				& target == ComponentLifeCycle.DEPLOYED) {
			isValid = true;
			msg += "------------ STOP ------------";
		} else {
			msg += "----- INVALID TRANSITION -----";
		}

		LOGGER.info(msg);

		return isValid;
	}

	private boolean shouldFailuresBeErased(ComponentLifeCycle source,
			ComponentLifeCycle target) {
		boolean erase = false;
		// STOP
		if (source == ComponentLifeCycle.DEPLOYED
				& target == ComponentLifeCycle.STARTED) {
			erase = true;
		}
		// UNDEPLOY's
		if (source == ComponentLifeCycle.NOT_SUPPORTED
				& target == ComponentLifeCycle.UNDEPLOYED) {
			erase = true;
		}
		return erase;
	}

}
