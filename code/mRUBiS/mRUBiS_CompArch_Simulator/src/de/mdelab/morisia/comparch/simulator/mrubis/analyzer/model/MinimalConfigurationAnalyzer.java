package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.InterfaceType;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.ModelAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisNames;

/**
 * Analyzes for the minimal configuration of each tenant's sub-architecture.
 * 
 * Each business interface should only be provided by exactly one started
 * component within one tenant. Otherwise, more than one component providing the
 * same functionality are running and consuming resources.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class MinimalConfigurationAnalyzer implements ModelAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(MinimalConfigurationAnalyzer.class.getName());

	/**
	 * Constructor.
	 */
	public MinimalConfigurationAnalyzer() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Minimal Configuration Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes for the minimal configuration of each tenant's sub-architecture.";
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
			Map<InterfaceType, List<Component>> provInterfaceComp = new HashMap<InterfaceType, List<Component>>();

			boolean componentInStateDeployed = false;
			boolean componentInStateNotSupported = false;

			for (Component eComponent : eTenant.getComponents()) {
				if (eComponent.getState() == ComponentLifeCycle.STARTED) {
					LOGGER.finer("[OK] Component \"" + eComponent.getUid()
							+ "\" is in state STARTED.");
					for (ProvidedInterface eProInterface : eComponent
							.getProvidedInterfaces()) {
						InterfaceType eProInterfaceType = eProInterface
								.getType();
						List<Component> eComponents = provInterfaceComp
								.get(eProInterfaceType);
						if (eComponents == null) {
							eComponents = new LinkedList<Component>();
							provInterfaceComp.put(eProInterfaceType,
									eComponents);
						}
						eComponents.add(eComponent);
					}
				} else if (eComponent.getState() == ComponentLifeCycle.DEPLOYED) {
					LOGGER.severe("[FAILURE] The component \""
							+ eComponent.getUid() + "\" is in state DEPLOYED.");
					failureCount++;
					componentInStateDeployed = true;
				} else if (eComponent.getState() == ComponentLifeCycle.NOT_SUPPORTED) {
					LOGGER.severe("[FAILURE] The component \""
							+ eComponent.getUid()
							+ "\" is in state NOT_SUPPORTED.");
					failureCount++;
					componentInStateNotSupported = true;
				}
			} // next component

			// logging
			if (!componentInStateDeployed) {
				LOGGER.info("[OK] There is no component in state DEPLOYED.");
			}
			if (!componentInStateNotSupported) {
				LOGGER.info("[OK] There is no component in state NOT_SUPPORTED.");
			}
			if (!componentInStateDeployed & !componentInStateNotSupported) {
				LOGGER.info("[OK] All components are in state STARTED.");
			}

			int currentNumberOfFailures = failureCount;
			for (InterfaceType eInterfaceType : provInterfaceComp.keySet()) {
				if (!eInterfaceType.getFullyQualifiedName().equals(
						MrubisNames.ITEM_FILTER_SERVICE_INTERFACE)) {
					// no filter component
					List<Component> eComponents = provInterfaceComp
							.get(eInterfaceType);
					if (eComponents.size() == 1) {
						LOGGER.finer("[OK] The interface \""
								+ eInterfaceType.getFullyQualifiedName()
								+ "\" is provided by one started component (\""
								+ eComponents.get(0).getUid() + "\").");
					} else {
						String severeMsg = "[FAILURE] The interface \""
								+ eInterfaceType.getFullyQualifiedName()
								+ "\" is provided by " + eComponents.size()
								+ " started components:";
						for (Component c : eComponents) {
							severeMsg = severeMsg + "\n\t\t --" + c.getUid();
						}
						LOGGER.severe(severeMsg);
						failureCount++;
					}
				}
			}
			// log summary
			int numberOfFailures = failureCount - currentNumberOfFailures;
			if (numberOfFailures == 0) {
				LOGGER.info("[OK] Each interface is provided by one started component.");
			} else {
				LOGGER.severe("[FAILURES] "
						+ numberOfFailures
						+ " interfaces are provided by more than one started component.");
			}

			// list not used component types of the tenant
			List<ComponentType> types = new LinkedList<ComponentType>(
					eArchitecture.getComponentTypes());
			for (Component c : eTenant.getComponents()) {
				types.remove(c.getType());
			}
			String msg = "[INFO] The following component types are not used:";
			if (types.size() == 0) {
				msg += "\n\t None";
			} else {
				for (ComponentType ct : types) {
					msg += "\n\t" + ct.getName();
				}
			}
			LOGGER.info(msg + "\n");

		} // next tenant

		return failureCount;
	}

}
