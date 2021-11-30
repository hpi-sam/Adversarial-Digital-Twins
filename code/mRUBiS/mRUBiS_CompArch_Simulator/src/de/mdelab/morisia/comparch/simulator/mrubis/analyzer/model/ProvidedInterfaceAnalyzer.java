package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model;

import java.util.HashSet;
import java.util.Set;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.ModelAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisNames;

/**
 * Analyzes each tenant whether it properly provides all business interfaces
 * such that each tenant (shop) is operable.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class ProvidedInterfaceAnalyzer implements ModelAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(ProvidedInterfaceAnalyzer.class.getName());

	/**
	 * Constructor.
	 */
	public ProvidedInterfaceAnalyzer() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Provided Interface Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes each tenant whether it properly provides all "
				+ "business interfaces such that each tenant (shop) is operable.";
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
			Set<String> allInterfaces = this.getAllFQInterfaceNames();
			for (Component eComponent : eTenant.getComponents()) {
				if (eComponent.getState() == ComponentLifeCycle.STARTED) {
					for (ProvidedInterface providedInterface : eComponent
							.getProvidedInterfaces()) {
						String fqInterfaceName = providedInterface.getType()
								.getFullyQualifiedName();
						if (!fqInterfaceName
								.equals(MrubisNames.ITEM_FILTER_SERVICE_INTERFACE)) {
							LOGGER.finer("[OK] Interface \"" + fqInterfaceName
									+ "\" is provided by \""
									+ eComponent.getUid() + "\" ("
									+ eComponent.getState() + ")");
							allInterfaces.remove(fqInterfaceName);
						}
					}
				}
			}

			if (allInterfaces.size() == 0) {
				LOGGER.info("[OK] All necessary interfaces are provided by a started component.\n");
			} else {
				String msg = "[FAILURES] The following interfaces are not provided by a started component:\n";
				for (String interfaceName : allInterfaces) {
					msg += "- [FAILURE] \"" + interfaceName + "\"\n";
					failureCount++;
				}
				LOGGER.severe(msg + "\n");
			}
		}
		return failureCount;
	}

	/**
	 * Returns the set of fully qualified names of business interfaces that must
	 * be properly provided by each tenant.
	 * 
	 * @return the set of fully qualified names of business interfaces that must
	 *         be properly provided by each tenant.
	 */
	private Set<String> getAllFQInterfaceNames() {
		Set<String> allInterfaces = new HashSet<String>();
		allInterfaces.add(MrubisNames.ABOUT_ME_SERIVICE_INTERFACE);
		allInterfaces.add(MrubisNames.VIEW_USER_INFO_SERVICE_INTERFACE);
		allInterfaces.add(MrubisNames.BROWSE_REGION_SERVICE_INTERFACE);
		allInterfaces.add(MrubisNames.USER_REGISTRATION_SERVICE_INTERFACE);
		allInterfaces.add(MrubisNames.BROWSE_CATEGORIES_SERVICE_INTERFACE);
		allInterfaces.add(MrubisNames.ITEM_REGISTRATION_SERVICE_INTERFACE);
		allInterfaces.add(MrubisNames.AUTHENTICATION_SERVICE_INTERFACE);
		allInterfaces.add(MrubisNames.BID_SERVICE_INTERFACE);
		allInterfaces.add(MrubisNames.BUY_NOW_SERVICE_INTERFACE);
		allInterfaces.add(MrubisNames.INVENTORY_SERVICE_INTERFACE);
		allInterfaces.add(MrubisNames.REPUTATION_SERVICE_INTERFACE);
		allInterfaces.add(MrubisNames.BO_PERSISTENCE_SERVICE_INTERFACE);
		allInterfaces.add(MrubisNames.BASIC_QUERY_SERVICE_INTERFACE);
		allInterfaces.add(MrubisNames.QUERY_SERVICE_INTERFACE);
		return allInterfaces;
	}

}
