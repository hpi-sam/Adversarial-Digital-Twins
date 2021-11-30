package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model;

import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.PerformanceStats;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.ModelAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisModelQuery;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisNames;

/**
 * Analyzes the average response time for the method
 * {@link MrubisNames#BROWSE_CATEGORIES_SERVICE_INTERFACE}.
 * {@link MrubisNames#GET_PERSONALIZED_ITEMS_SIGNATURE} of each tenant based on
 * the corresponding {@link PerformanceStats} elements. Ideally, the average
 * response time should be around a response time goal between a lower and an
 * upper boundary.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class AverageResponseTimeAnalyzer implements ModelAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(AverageResponseTimeAnalyzer.class.getName());

	private int responseTimeUpperThreshold;
	private int responseTimeLowerThreshold;
	private int responseTimeGoal;

	/**
	 * Constructor.
	 * 
	 * @param responseTimeUpperThreshold
	 *            the upper threshold (boundary) of the average response time.
	 * @param responseTimeLowerThreshold
	 *            the lower threshold (boundary) of the average response time.
	 * @param responseTimeGoal
	 *            the goal (target value) for the average response time
	 */
	public AverageResponseTimeAnalyzer(int responseTimeUpperThreshold,
			int responseTimeLowerThreshold, int responseTimeGoal) {
		this.responseTimeUpperThreshold = responseTimeUpperThreshold;
		this.responseTimeLowerThreshold = responseTimeLowerThreshold;
		this.responseTimeGoal = responseTimeGoal;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Average Response Time Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes the average response time for the method "
				+ MrubisNames.BROWSE_CATEGORIES_SERVICE_INTERFACE + "."
				+ MrubisNames.GET_PERSONALIZED_ITEMS_SIGNATURE
				+ " of each tenant whether it is located between "
				+ this.responseTimeLowerThreshold + " and "
				+ this.responseTimeUpperThreshold
				+ " around the target value of " + this.responseTimeGoal;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int analyze(Architecture eArchitecture) {
		int failureCount = 0;
		int numberOfAvailableFilterTypes = MrubisModelQuery.INSTANCE
				.getNumberOfAvailableFilterTypes(eArchitecture);

		for (Tenant tenant : eArchitecture.getTenants()) {
			LOGGER.info("Analyzing the tenant \"" + tenant.getName() + "\" ...");

			try {
				PerformanceStats targetMethodStats = MrubisModelQuery.INSTANCE
						.getPerformanceStats(tenant);

				long totalTime = targetMethodStats.getTotalTime();
				long invocationCount = targetMethodStats.getInvocationCount();
				long avgResponseTime = totalTime / invocationCount;

				int numberOfFilters = MrubisModelQuery.INSTANCE
						.getAllFiltersOfTenant(tenant).size();

				if (avgResponseTime > this.responseTimeUpperThreshold) {
					// exceeding the upper threshold is considered as a
					// failure if the size of the pipe is > 1.
					if (numberOfFilters > 1) {
						LOGGER.severe("[FAILURE] The average response time is "
								+ avgResponseTime
								+ " and therefore greater than "
								+ this.responseTimeUpperThreshold
								+ " and the pipe has " + numberOfFilters
								+ " filters.\n");
						failureCount++;
					} else {
						LOGGER.warning("[WARNING] The average response time is "
								+ avgResponseTime
								+ " and therefore greater than "
								+ this.responseTimeUpperThreshold
								+ " but the pipe has only "
								+ numberOfFilters
								+ " filters.\n");
					}
				} else if (avgResponseTime < this.responseTimeLowerThreshold) {
					// this is only a failure if less than the available filters
					// are employed in the pipe. Hence, if the pipe does not
					// contain a filter of each type, then it is is failure.

					if (numberOfFilters == numberOfAvailableFilterTypes) {
						// each filter type is instantiated and used at least
						// once in the pipe. Whether a type is instantiated more
						// than once is checked by the method
						// this#checkFilterTypes
						// Hence, the pipe has the maximal length.
						LOGGER.info("[OK] The average response time is "
								+ avgResponseTime
								+ " and therefore smaller than "
								+ this.responseTimeLowerThreshold
								+ " but the pipe already has the maximal length of "
								+ numberOfAvailableFilterTypes + " filters.\n");
					} else if (numberOfFilters < numberOfAvailableFilterTypes) {
						// The pipe does not have the maximal length.
						LOGGER.severe("[FAILURE] The average response time is "
								+ avgResponseTime
								+ " and therefore smaller than "
								+ this.responseTimeLowerThreshold
								+ " while the pipe with its "
								+ numberOfFilters
								+ " filters does not have the maximal length of "
								+ numberOfAvailableFilterTypes + " filters.\n");
						failureCount++;
					} else {
						// numberOfFilters > numberOfAvailableFilterTypes
						LOGGER.severe("[FAILURE] The average response time is "
								+ avgResponseTime
								+ " and therefore smaller than "
								+ this.responseTimeLowerThreshold
								+ " while the pipe with its "
								+ numberOfFilters
								+ " filters has more than "
								+ numberOfAvailableFilterTypes
								+ " filters. There is additionally something wrong with the pipe configuration.\n");
						failureCount++;
					}

				} else {
					LOGGER.info("[OK] The average response time is "
							+ avgResponseTime
							+ " and therefore acceptable with respect to the goal of "
							+ this.responseTimeGoal + ". (The pipe has "
							+ numberOfFilters + " filters).\n");
				}
			} catch (Exception e) {
				LOGGER.severe("Simulator cannot analyze the average response time to check the runtime model: "
						+ e.getMessage());
			}

		} // next tenant

		return failureCount;
	}

}
