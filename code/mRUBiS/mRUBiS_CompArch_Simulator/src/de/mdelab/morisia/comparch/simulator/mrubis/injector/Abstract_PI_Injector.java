package de.mdelab.morisia.comparch.simulator.mrubis.injector;

import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.MethodSpecification;
import de.mdelab.morisia.comparch.PerformanceStats;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisModelQuery;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisNames;

/**
 * Abstract injector for PI2 and PI3.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class Abstract_PI_Injector {

	private final static Logger LOGGER = Logger
			.getLogger(Abstract_PI_Injector.class.getName());

	private int performanceMonitoringInterval;

	/**
	 * Constructor.
	 * 
	 * @param performanceMonitoringInterval
	 *            interval in terms of invocation counts between two performance
	 *            measurements.
	 */
	protected Abstract_PI_Injector(int performanceMonitoringInterval) {
		this.performanceMonitoringInterval = performanceMonitoringInterval;
	}

	/**
	 * Changes the average response time of the method
	 * {@code MrubisNames#GET_PERSONALIZED_ITEMS_SIGNATURE} of the interface
	 * {@code MrubisNames#BROWSE_CATEGORIES_SERVICE_INTERFACE} provided by
	 * exactly one component of the given tenant. Thus, the corresponding
	 * {@code PerformanceStats} element is updated by setting new values to its
	 * attributes such that the average response time is
	 * <code>avgResponseTime</code>.
	 * 
	 * 
	 * @param target
	 *            the give tenant
	 * @param avgResponseTime
	 *            the average response time to be set
	 * @return <code>true</code> if the average response could be successfully
	 *         updated, otherwise<code>false</code>.
	 */
	protected boolean changeResponseTime(Tenant tenant, long avgResponseTime) {

		// find the provided interface
		// "de.hpi.sam.rubis.itemmgmt.BrowseCategoriesService"
		ProvidedInterface browseCategoriesInterface = MrubisModelQuery.INSTANCE
				.getProvidedInterfaceByName(tenant,
						MrubisNames.BROWSE_CATEGORIES_SERVICE_INTERFACE);
		if (browseCategoriesInterface == null) {
			LOGGER.warning(" -- The provided interface \""
					+ MrubisNames.BROWSE_CATEGORIES_SERVICE_INTERFACE
					+ "\" cannot be found for the tenant " + tenant.getName());
			return false;
		} else {
			// get component providing the interface
			Component itemMgmtService = browseCategoriesInterface
					.getComponent();

			// find method
			// "getPersonalizedItems(java.lang.String,java.lang.String)"
			MethodSpecification targetMethod = MrubisModelQuery.INSTANCE
					.findMethodSpecification(browseCategoriesInterface,
							MrubisNames.GET_PERSONALIZED_ITEMS_SIGNATURE);
			if (targetMethod == null) {
				LOGGER.warning(" -- The method \""
						+ MrubisNames.GET_PERSONALIZED_ITEMS_SIGNATURE
						+ "\" of the interface of type \""
						+ browseCategoriesInterface.getType()
								.getFullyQualifiedName()
						+ "\" cannot be found for component \""
						+ itemMgmtService.getUid() + "\" in the tenant "
						+ tenant.getName());
				return false;
			} else {
				// find performance stats element for the method
				PerformanceStats targetMethodStats = MrubisModelQuery.INSTANCE
						.getStatsForMethod(browseCategoriesInterface,
								targetMethod);
				if (targetMethodStats == null) {
					LOGGER.warning(" -- The are no performance stats for the method \""
							+ MrubisNames.GET_PERSONALIZED_ITEMS_SIGNATURE
							+ "\" of the interface of type \""
							+ browseCategoriesInterface.getType()
									.getFullyQualifiedName()
							+ "\" for component \""
							+ itemMgmtService.getUid()
							+ "\" in the tenant " + tenant.getName());
					return false;
				} else {
					// update the average response time by updating the
					// totalTime and the invocationCount of the performance
					// stats element
					long totalTime = targetMethodStats.getTotalTime();
					long invocationCount = targetMethodStats
							.getInvocationCount();
					long old_avgResponseTime = totalTime / invocationCount;

					long new_invocationCount = invocationCount
							+ this.performanceMonitoringInterval;
					long new_totalTime = new_invocationCount * avgResponseTime;

					targetMethodStats.setTotalTime(new_totalTime);
					targetMethodStats.setInvocationCount(new_invocationCount);
					LOGGER.info(" -- For the tenant \"" + tenant.getName()
							+ "\", the performance stats of method \""
							+ targetMethod.getSignature()
							+ "\" of interface \""
							+ browseCategoriesInterface.getUid()
							+ "\" of component \"" + itemMgmtService.getUid()
							+ "\" has been changed as follows:\n"
							+ "Total time:\t\t" + totalTime + " => "
							+ new_totalTime + "\n" + "Invocation Count:\t"
							+ invocationCount + " => " + new_invocationCount
							+ "\n" + "Average response time:\t"
							+ old_avgResponseTime + " => " + avgResponseTime);

					// scale all other performance stats of this tenant
					// linearly such that the average response time
					// remain the same for these other stats elements.
					this.scaleOtherPerformanceStats(tenant, targetMethodStats);
					return true;
				}
			}
		}

	}

	/**
	 * Scale all other performance stats of this tenant linearly such that the
	 * average response time remains the same for these other stats elements.
	 * 
	 * @param tenant
	 *            the tenant
	 * @param targetMethodStats
	 *            the performace stats element of the method to be excluded from
	 *            the scaling
	 */
	private void scaleOtherPerformanceStats(Tenant tenant,
			PerformanceStats targetMethodStats) {

		for (Component comp : tenant.getComponents()) {
			for (ProvidedInterface pi : comp.getProvidedInterfaces()) {
				for (PerformanceStats ps : pi.getPerformanceStats()) {
					if (!ps.equals(targetMethodStats)) {
						// only scale stats of methods that are used
						if (ps.getInvocationCount() > 0) {
							long ps_avgResponseTime = ps.getTotalTime()
									/ ps.getInvocationCount();
							ps.setInvocationCount(ps.getInvocationCount()
									+ this.performanceMonitoringInterval);
							ps.setTotalTime(ps_avgResponseTime
									* ps.getInvocationCount());
						}
					}
				}
			}
		}
	}

}
