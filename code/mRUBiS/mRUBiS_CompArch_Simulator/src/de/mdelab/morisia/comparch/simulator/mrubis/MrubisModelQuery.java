package de.mdelab.morisia.comparch.simulator.mrubis;

import java.util.LinkedList;
import java.util.List;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.Connector;
import de.mdelab.morisia.comparch.InterfaceType;
import de.mdelab.morisia.comparch.MethodSpecification;
import de.mdelab.morisia.comparch.MonitoredProperty;
import de.mdelab.morisia.comparch.Parameter;
import de.mdelab.morisia.comparch.PerformanceStats;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.RequiredInterface;
import de.mdelab.morisia.comparch.Tenant;

/**
 * Reusable and stateless queries of the model.
 * 
 * @author thomas vogel
 * @version 0.21
 * 
 */
public class MrubisModelQuery {

	/**
	 * The singleton instance of the class.
	 */
	public final static MrubisModelQuery INSTANCE = new MrubisModelQuery();

	private final static Logger LOGGER = Logger
			.getLogger(MrubisModelQuery.class.getName());

	/**
	 * Returns all filter components of the given tenant.
	 * 
	 * @param tenant
	 *            the given tenant
	 * @return ordered list of filters; the order is the same as the order of
	 *         the filters in the pipe of the tenant.
	 */
	public List<Component> getAllFiltersOfTenant(Tenant tenant) {
		Component queryService = null;
		OUTERLOOP: for (Component c : tenant.getComponents()) {
			for (ProvidedInterface pi : c.getProvidedInterfaces()) {
				if (pi.getType().getFullyQualifiedName()
						.equals(MrubisNames.BASIC_QUERY_SERVICE_INTERFACE)) {
					queryService = c;
					break OUTERLOOP;
				}
			}
		}

		List<Component> filters = new LinkedList<>();

		if (queryService == null) {
			LOGGER.warning("Filters cannot be obtained since "
					+ "the \"Query Service\" cannot be found for the tenant \""
					+ tenant.getName() + "\".");
			return filters;
		} else {

			// -- get first filter component
			Component filter = null;
			for (RequiredInterface reqInterface : queryService
					.getRequiredInterfaces()) {
				if (reqInterface.getType().getFullyQualifiedName()
						.equals(MrubisNames.ITEM_FILTER_SERVICE_INTERFACE)) {
					Connector c = reqInterface.getConnector();
					if (c != null && c.getTarget() != null) {
						filter = c.getTarget().getComponent();
						break;
					} else {
						LOGGER.warning("Filters cannot be obtained since "
								+ "the \"Query Service\" is not connected to any filter in the tenant \""
								+ queryService.getTenant().getName() + "\".");
						return filters;
					}
				}
			}
			filters.add(filter);
			// -- get iteratively the next filter
			while (filter != null) {
				if (filter.getRequiredInterfaces().size() > 0) {
					Connector c = filter.getRequiredInterfaces().get(0)
							.getConnector();
					if (c != null) {
						if (c.getTarget() != null) {
							Component nextFilter = c.getTarget().getComponent();
							if (nextFilter.equals(filter)) {
								LOGGER.warning("Filter \""
										+ filter.getUid()
										+ "\" is connected to itself. This is an invalid configuration of the pipe.");
								return filters;
							}
							if (filters.contains(nextFilter)) {
								// the filter has already been collected and
								// thus it
								// has been visited more than one time.

								// create log message
								String msg = "";
								int pos = 1;
								for (Component f : filters) {
									msg = msg + "    (" + pos + ") "
											+ f.getUid() + "\n";
									pos++;
								}
								msg = msg + "    (" + pos + ") "
										+ nextFilter.getUid();
								LOGGER.severe("The pipe contains a cycle among the filters. This is an invalid configuration of the pipe:\n"
										+ msg);
								return filters;
							}
							filters.add(nextFilter);
							// next iteration
							filter = nextFilter;
						} else {
							// c.getTarget() == null
							filter = null;
						}

					} else {
						// c == null
						filter = null;
					}
				} else {
					filter = null;
				}
			}
			return filters;
		}
	}

	/**
	 * Returns the first occurrence of a provided interface with the given
	 * fullyQualifiedInterfaceName in the given tenant.
	 * 
	 * Should be used for provided interface types that are only provided once
	 * in a tenant. Hence, it does not work for ItemFilter interfaces that are
	 * provided by multiple filter components in one tenant.
	 * 
	 * @param eTenant
	 *            the given tenant
	 * @param fullyQualifiedInterfaceName
	 *            the given fully qualified interface name
	 * @return the corresponding firstly found provided interface
	 */
	public ProvidedInterface getProvidedInterfaceByName(Tenant eTenant,
			String fullyQualifiedInterfaceName) {
		ProvidedInterface providedInterface = null;
		OUTERLOOP: for (Component c : eTenant.getComponents()) {
			for (ProvidedInterface pi : c.getProvidedInterfaces()) {
				if (pi.getType().getFullyQualifiedName()
						.equals(fullyQualifiedInterfaceName)) {
					providedInterface = pi;
					break OUTERLOOP;
				}
			}
		}
		return providedInterface;
	}

	/**
	 * Returns the specification of the method belonging to the given provided
	 * interface and having the given signature.
	 * 
	 * @param pi
	 *            the given provided interface
	 * @param signature
	 *            the given signature
	 * @return the method specification
	 */
	public MethodSpecification findMethodSpecification(ProvidedInterface pi,
			String signature) {
		MethodSpecification result = null;
		for (MethodSpecification m : pi.getType().getMethodSpecification()) {
			if (m.getSignature().equals(signature)) {
				result = m;
				break;
			}
		}
		return result;
	}

	/**
	 * Returns the performance statistics of the given method that belongs to
	 * the given provided interface.
	 * 
	 * @param providedInterface
	 *            the given provided interface.
	 * @param method
	 *            the given method.
	 * @return the performance statistics of the method.
	 */
	public PerformanceStats getStatsForMethod(
			ProvidedInterface providedInterface, MethodSpecification method) {
		PerformanceStats stats = null;
		for (PerformanceStats s : providedInterface.getPerformanceStats()) {
			if (s.getMethod().equals(method)) {
				stats = s;
				break;
			}
		}
		return stats;
	}

	/**
	 * Returns the parameter belonging to the given component and having the
	 * given name.
	 * 
	 * @param component
	 *            the given component
	 * @param paramName
	 *            the given parameter name
	 * @return the corresponding parameter
	 */
	public Parameter getParameterByName(Component component, String paramName) {
		Parameter param = null;
		for (Parameter p : component.getParameters()) {
			if (p.getType().getName().equals(paramName)) {
				param = p;
				break;
			}
		}
		return param;
	}

	/**
	 * Returns the monitored property belonging to the given component and
	 * having the given name.
	 * 
	 * @param component
	 *            the given component
	 * @param propName
	 *            the given property name
	 * @return the corresponding monitored property
	 */
	public MonitoredProperty getMonitoredPropertyByName(Component component,
			String propName) {
		MonitoredProperty prop = null;
		for (MonitoredProperty p : component.getMonitoredProperties()) {
			if (p.getName().equals(propName)) {
				prop = p;
				break;
			}
		}
		return prop;
	}

	/**
	 * Returns the slope of the given filter component.
	 * 
	 * @param filter
	 *            the given filter component
	 * @return the slope of the given filter component
	 */
	public double getSlopeOfFilter(Component filter) {
		MonitoredProperty s = this.getMonitoredPropertyByName(filter,
				MrubisNames.SELECTION_RATE_PROPERTY_NAME);
		double s_value = Double.parseDouble(s.getValue());
		MonitoredProperty c = this.getMonitoredPropertyByName(filter,
				MrubisNames.LOCAL_COMPUTATION_TIME_PROPERTY_NAME);
		long c_value = Long.parseLong(c.getValue());
		return s_value / c_value;
	}

	/**
	 * Returns all filter component types available in the model.
	 * 
	 * @param architecture
	 *            the root element of the model.
	 * @return list of all filter types available in the model. A filter type is
	 *         a component type that provided the interface of type
	 *         <code>de.hpi.sam.rubis.filter.ItemFilter</code>.
	 */
	public List<ComponentType> getAllFilterTypes(Architecture architecture) {
		List<ComponentType> filterTypes = new LinkedList<>();
		for (ComponentType ct : architecture.getComponentTypes()) {
			for (InterfaceType it : ct.getProvidedInterfaceTypes()) {
				if (it.getFullyQualifiedName().equals(
						MrubisNames.ITEM_FILTER_SERVICE_INTERFACE)) {
					filterTypes.add(ct);
					break; // break inner loop and check next component type
				}
			}
		}
		return filterTypes;

	}

	/**
	 * Returns the number of filter component types available in the model.
	 * 
	 * @param architecture
	 *            the root element of the model.
	 * @return the number of filter component types available in the model.
	 */
	public int getNumberOfAvailableFilterTypes(Architecture architecture) {
		return this.getAllFilterTypes(architecture).size();
	}

	/**
	 * Returns the performance statistics of the method
	 * {@code MrubisNames#GET_PERSONALIZED_ITEMS_SIGNATURE} belonging to the
	 * interface {@code MrubisNames#BROWSE_CATEGORIES_SERVICE_INTERFACE}
	 * provided by a component of the given tenant.
	 * 
	 * @param tenant
	 *            the given tenant.
	 * @return the corresponding performance statistics
	 */
	public PerformanceStats getPerformanceStats(Tenant tenant) {
		// find browseCategoriesInterface
		ProvidedInterface browseCategoriesInterface = MrubisModelQuery.INSTANCE
				.getProvidedInterfaceByName(tenant,
						MrubisNames.BROWSE_CATEGORIES_SERVICE_INTERFACE);
		// find method
		MethodSpecification targetMethod = MrubisModelQuery.INSTANCE
				.findMethodSpecification(browseCategoriesInterface,
						MrubisNames.GET_PERSONALIZED_ITEMS_SIGNATURE);
		// find performance stats
		PerformanceStats targetMethodStats = MrubisModelQuery.INSTANCE
				.getStatsForMethod(browseCategoriesInterface, targetMethod);
		return targetMethodStats;
	}
}
