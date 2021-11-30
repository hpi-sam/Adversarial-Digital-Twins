package de.mdelab.morisia.comparch.simulator.mrubis.injector;

import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.Injector;
import de.mdelab.morisia.comparch.simulator.IssueType;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisNames;

/**
 * Injects a {@link IssueType#PI3}.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class PI3_Injector extends Abstract_PI_Injector implements
		Injector<Tenant> {

	private int responseTimeLowerThreshold;

	/**
	 * Constructor.
	 * 
	 * @param performanceMonitoringIntervall
	 *            interval in terms of invocation counts between two performance
	 *            measurements.
	 * @param responseTimeLowerThreshold
	 *            lower threshold of the average response time. Injecting a PI3
	 *            sets the average response time to a value smaller than the
	 *            lower threshold.
	 */
	public PI3_Injector(int performanceMonitoringIntervall,
			int responseTimeLowerThreshold) {
		super(performanceMonitoringIntervall);
		this.responseTimeLowerThreshold = responseTimeLowerThreshold;
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "PI3 Injector";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Decreases the response time of the tenant's method \""
				+ MrubisNames.BROWSE_CATEGORIES_SERVICE_INTERFACE + "#"
				+ MrubisNames.GET_PERSONALIZED_ITEMS_SIGNATURE
				+ "\" below the threshold of "
				+ this.responseTimeLowerThreshold;
	}

	/**
	 * {@inheritDoc}
	 */
	public IssueType getIssueType() {
		return IssueType.PI3;
	}

	/**
	 * Injects a PI3 by decreasing the average response time of the method
	 * {@code MrubisNames#GET_PERSONALIZED_ITEMS_SIGNATURE} below the given
	 * threshold for the given tenant.
	 * 
	 * {@inheritDoc}
	 */
	@Override
	public boolean inject(Tenant tenant) {
		return super.changeResponseTime(tenant,
				this.responseTimeLowerThreshold - 50);
	}

}
