package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.ModelAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisModelQuery;

/**
 * Analyzes that each filter component of a tenant's pipe is of a different
 * filter component type. (It does not make sense to apply more than one filter
 * of the same type since they realize the same functionality).
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class FilterTypeAnalyzer implements ModelAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(FilterTypeAnalyzer.class.getName());

	/**
	 * Constructor.
	 */
	public FilterTypeAnalyzer() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Filter Type Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes that each filter component of a tenant's pipe is of a different filter component type."
				+ "(A failure is created if a pipe contains more than one filter of the same type.)";
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

			Map<ComponentType, Integer> filterTypes = new HashMap<>();
			Map<ComponentType, List<Component>> filterTypes2Filters = new HashMap<>();

			List<Component> eFilters = MrubisModelQuery.INSTANCE
					.getAllFiltersOfTenant(eTenant);
			for (Component eFilter : eFilters) {
				ComponentType eFilterType = eFilter.getType();

				Integer i = filterTypes.get(eFilterType);
				if (i == null || i.intValue() == 0) {
					filterTypes.put(eFilterType, 1);
				} else {
					filterTypes.put(eFilterType, i + 1);
				}

				List<Component> instances = filterTypes2Filters
						.get(eFilterType);
				if (instances == null) {
					instances = new LinkedList<>();
					filterTypes2Filters.put(eFilterType, instances);
				}
				instances.add(eFilter);

			}

			// LOG the pipe
			String debug = "Debug the pipe ...";
			for (ComponentType type : filterTypes2Filters.keySet()) {
				debug = debug + "\n-- Of the component type \"" + type.getUid()
						+ "\" the following components are part of the pipe:";
				List<Component> comps = filterTypes2Filters.get(type);
				for (Component c : comps) {
					debug = debug + "\n\t-- " + c.getUid();
				}
			}
			LOGGER.finer(debug);
			// LOG the pipe

			boolean isFailure = false;
			for (ComponentType filterType : filterTypes.keySet()) {
				int numberOfOccurences = filterTypes.get(filterType);
				if (numberOfOccurences >= 2) {
					isFailure = true;
					LOGGER.severe("[FAILURE] The pipe contains "
							+ numberOfOccurences
							+ " (hence, more than one) filters of type \""
							+ filterType.getUid() + "\".");
					failureCount++;
				}
			}

			if (!isFailure) {
				LOGGER.info("[OK] All filters of the pipe are of a different type.\n");
			}

		} // next tenant

		return failureCount;
	}

}
