package de.mdelab.morisia.comparch.simulator.mrubis.injector;

import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.Injector;
import de.mdelab.morisia.comparch.simulator.IssueType;
import de.mdelab.morisia.comparch.simulator.impl.SimulatorUtil;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisNames;

/**
 * Injects a {@link IssueType#PI2}.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class PI2_Injector extends Abstract_PI_Injector implements
		Injector<Tenant> {

	private int responseTimeUpperThreshold;

	/**
	 * Constructor.
	 * 
	 * @param performanceMonitoringIntervall
	 *            interval in terms of invocation counts between two performance
	 *            measurements.
	 * @param responseTimeUpperThreshold
	 *            upper threshold of the average response time. Injecting a PI2
	 *            sets the average response time to a value greater than the
	 *            upper threshold.
	 */
	public PI2_Injector(int performanceMonitoringIntervall,
			int responseTimeUpperThreshold) {
		super(performanceMonitoringIntervall);
		this.responseTimeUpperThreshold = responseTimeUpperThreshold;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "PI2 Injector";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Increases the response time of the tenant's method \""
				+ MrubisNames.BROWSE_CATEGORIES_SERVICE_INTERFACE + "#"
				+ MrubisNames.GET_PERSONALIZED_ITEMS_SIGNATURE
				+ "\" above the threshold of "
				+ this.responseTimeUpperThreshold;
	}

	/**
	 * {@inheritDoc}
	 */
	public IssueType getIssueType() {
		return IssueType.PI2;
	}

	/**
	 * Injects a PI2 by increasing the average response time of the method
	 * {@code MrubisNames#GET_PERSONALIZED_ITEMS_SIGNATURE} above the given
	 * threshold for the given tenant.
	 * 
	 * {@inheritDoc}
	 */
	@Override
	public boolean inject(Tenant tenant) {
		return super.changeResponseTime(tenant, this.responseTimeUpperThreshold
				+ SimulatorUtil.randInt(50, 150));
	}

}
