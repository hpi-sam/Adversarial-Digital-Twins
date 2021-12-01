package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.model;

import java.util.List;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.ModelAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisModelQuery;

/**
 * Analyzes the size of the pipe of filters for each tenant.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class PipeSizeAnalyzer implements ModelAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(PipeSizeAnalyzer.class.getName());

	private int minimalPipeSize;

	/**
	 * Constructor.
	 * 
	 * @param minimalPipeSize
	 *            the minimal size a pipe should have in terms of number of
	 *            filters.
	 */
	public PipeSizeAnalyzer(int minimalPipeSize) {
		this.minimalPipeSize = minimalPipeSize;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Pipe Size Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes whether the pipe of filters contains at least "
				+ this.minimalPipeSize
				+ " filters. (Only warnings are created if the pipe has less than "
				+ this.minimalPipeSize + " filters.)";
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
			List<Component> filters = MrubisModelQuery.INSTANCE
					.getAllFiltersOfTenant(eTenant);
			if (filters.size() == 0) {
				LOGGER.warning("The size of the pipe cannot be checked.\n");
			} else if (filters.size() < this.minimalPipeSize) {
				LOGGER.warning("[WARNING] The pipe contains only "
						+ filters.size() + " filters, which is less than "
						+ this.minimalPipeSize + ".\n");
			} else {
				LOGGER.info("[OK] The pipe contains "
						+ filters.size()
						+ " filters, which are sufficiently enough filters, that is, "
						+ this.minimalPipeSize + " or more.\n");
			}
		} // next tenant

		return failureCount;
	}

}
