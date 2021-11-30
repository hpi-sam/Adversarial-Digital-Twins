package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model;

import java.util.List;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.ModelAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisModelQuery;

/**
 * Analyzes for each tenant's pipe whether the filters in the pipe are
 * well-ordered based on their slopes of selection rate and local computation
 * time.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class FilterOrderAnalyzer implements ModelAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(FilterOrderAnalyzer.class.getName());

	/**
	 * Constructor.
	 */
	public FilterOrderAnalyzer() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Filter Order Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes for each tenant's pipe whether the filters in the pipe are "
				+ "well-ordered based on their slopes of selection rate and local computation time.";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int analyze(Architecture eArchitecture) {
		int failureCount = 0;

		for (Tenant tenant : eArchitecture.getTenants()) {
			LOGGER.info("Analyzing the tenant \"" + tenant.getName() + "\" ...");

			int currentNumberOfFailures = failureCount;
			String report = "";

			List<Component> filters = MrubisModelQuery.INSTANCE
					.getAllFiltersOfTenant(tenant);
			if (filters.size() == 0) {
				LOGGER.warning("The order of filters cannot be checked.");
				return failureCount;
			}

			Double[] slopes = new Double[filters.size()];

			// calculate slopes
			for (int i = 0; i < filters.size(); i++) {
				Component filter = filters.get(i);
				double slope = MrubisModelQuery.INSTANCE
						.getSlopeOfFilter(filter);
				slopes[i] = slope;
			}

			// compare slopes
			for (int j = 0; j < slopes.length - 1; j++) {
				int position = j + 1;
				if ((slopes[j] - slopes[j + 1]) < 0) {
					LOGGER.severe("[FAILURE] The "
							+ position
							+ ". filter \""
							+ filters.get(j).getUid()
							+ "\" is not located at its optimal position in the pipe.\n"
							+ "Its slope is " + slopes[j]
							+ " and the slope of its subsequent filter is "
							+ slopes[j + 1]);
					failureCount++;
				}
				report = report + "Slope of the ";
				if (position < 10) {
					report = report + " ";
				}
				report = report + position + ". filter: " + slopes[j]
						+ "\t Filter: " + filters.get(j).getUid() + "\n";
			}
			// add last filter to the report
			report = report + "Slope of the ";
			if (slopes.length < 10) {
				report = report + " ";
			}
			report = report + slopes.length + ". filter: "
					+ slopes[slopes.length - 1] + "\t Filter: "
					+ filters.get(slopes.length - 1).getUid() + "\n";

			int numberOfFailuresInTenant = failureCount
					- currentNumberOfFailures;
			if (numberOfFailuresInTenant == 0) {
				LOGGER.info("[OK] All filters are well-ordered in the pipe.\n");
			} else {
				LOGGER.severe("[FAILURES] " + numberOfFailuresInTenant
						+ " filters are not well-ordered in the pipe:\n"
						+ report);
			}

		} // next tenant
		return failureCount;
	}

}
