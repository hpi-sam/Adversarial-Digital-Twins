package de.mdelab.morisia.comparch.simulator.mrubis.injector;

import java.util.List;
import java.util.logging.Logger;

import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.MonitoredProperty;
import de.mdelab.morisia.comparch.Parameter;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.Injector;
import de.mdelab.morisia.comparch.simulator.IssueType;
import de.mdelab.morisia.comparch.simulator.impl.SimulatorUtil;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisModelQuery;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisNames;

/**
 * Potentially injects a {@link IssueType#PI1}.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class PI1_Injector implements Injector<Tenant> {

	private final static Logger LOGGER = Logger.getLogger(PI1_Injector.class
			.getName());

	/**
	 * Constructor.
	 */
	public PI1_Injector() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "PI1 Injector";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Changes the selection rate and/or local computation time of a "
				+ "filter component such that the filter might not be located "
				+ "any more at an optimal position.";
	}

	/**
	 * {@inheritDoc}
	 */
	public IssueType getIssueType() {
		return IssueType.PI1;
	}

	/**
	 * Changes the MonitoredProperty selection rate and/or local computation
	 * time of a filer component belonging to the given tenant. The change may
	 * result in a PI1.
	 * 
	 * {@inheritDoc}
	 */
	@Override
	public boolean inject(Tenant tenant) {
			// get all filters of the given tenant
			List<Component> filters = MrubisModelQuery.INSTANCE
					.getAllFiltersOfTenant(tenant);
			if (filters.size() == 0) {
				LOGGER.warning(" -- Filter characteristics cannot be changed. "
						+ "No filter can be found starting from The \"Query Service\" in the tenant "
						+ tenant.getName());
				return false;
			} else {
				// select one random filter of the tenant's filters
				int index = SimulatorUtil.randInt(0, filters.size() - 1);
				Component selectedFilter = filters.get(index);

				// get thresholds of the selected filter
				Parameter s_threshold_Param = MrubisModelQuery.INSTANCE
						.getParameterByName(
								selectedFilter,
								MrubisNames.SELECTION_RATE_THRESHOLD_PARAMETER_NAME);
				double s_threshold = Double.parseDouble(s_threshold_Param
						.getValue());
				Parameter c_threshold_Param = MrubisModelQuery.INSTANCE
						.getParameterByName(
								selectedFilter,
								MrubisNames.LOCAL_COMPUTATION_TIME_THRESHOLD_PARAMETER_NAME);
				double c_threshold = Double.parseDouble(c_threshold_Param
						.getValue());

				// get MonitoredProperties of the selected filter
				MonitoredProperty s = MrubisModelQuery.INSTANCE
						.getMonitoredPropertyByName(selectedFilter,
								MrubisNames.SELECTION_RATE_PROPERTY_NAME);
				double s_value = Double.parseDouble(s.getValue());
				MonitoredProperty c = MrubisModelQuery.INSTANCE
						.getMonitoredPropertyByName(
								selectedFilter,
								MrubisNames.LOCAL_COMPUTATION_TIME_PROPERTY_NAME);
				long c_value = Long.parseLong(c.getValue());
				if (s_value == 0) {
					s_value = 0.01;
				}
				if (c_value == 0) {
					c_value = 30;
				}

				// change s or c by thresholds or a multitude of the
				// thresholds
				int changeType = SimulatorUtil.randInt(0, 1);

				// change s or c by thresholds
				if (changeType == 0) {

					// change s or c or both
					int whatToChange = SimulatorUtil.randInt(0, 2);
					// change s
					if (whatToChange == 0) {
						// increase or decrease s
						int increase = SimulatorUtil.randInt(0, 1);
						if (increase == 0) {
							// increase s
							this.increaseSelectionRateByThreshold(
									selectedFilter, s, s_value, s_threshold,
									tenant);
						} else {
							// decrease s
							this.decreaseSelectionRateByThreshold(
									selectedFilter, s, s_value, s_threshold,
									tenant);
						}

					} else // change c
					if (whatToChange == 1) {

						// increase or decrease c
						int increase = SimulatorUtil.randInt(0, 1);
						if (increase == 0) {
							// increase c
							this.increaseComputationTimeByThreshold(
									selectedFilter, c, c_value, c_threshold,
									tenant);
						} else {
							// decrease c
							this.decreaseComputationTimeByThreshold(
									selectedFilter, c, c_value, c_threshold,
									tenant);
						}

					} else // change s and c
					if (whatToChange == 2) {

						// increase or decrease s
						int increase = SimulatorUtil.randInt(0, 1);
						if (increase == 0) {
							// increase s
							this.increaseSelectionRateByThreshold(
									selectedFilter, s, s_value, s_threshold,
									tenant);
						} else {
							// decrease s
							this.decreaseSelectionRateByThreshold(
									selectedFilter, s, s_value, s_threshold,
									tenant);
						}

						// increase or decrease c
						increase = SimulatorUtil.randInt(0, 1);
						if (increase == 0) {
							// increase c
							this.increaseComputationTimeByThreshold(
									selectedFilter, c, c_value, c_threshold,
									tenant);
						} else {
							// decrease c
							this.decreaseComputationTimeByThreshold(
									selectedFilter, c, c_value, c_threshold,
									tenant);
						}

					}

				} else {
					// change s by a multitude of the thresholds
					// change c by the threshold
					double scalar = this.calculateScalarForTheSelectionRate(
							filters, selectedFilter, c_value, s_value);
					double new_s_value = s_value * scalar;
					s.setValue("" + new_s_value);

					if (scalar >= 1) {
						// increased s
						LOGGER.info(" -- Selection rate of filter \""
								+ selectedFilter.getUid() + "\" in tenant \""
								+ tenant.getName()
								+ "\" has been increased from " + s_value
								+ " to " + new_s_value);

						// decrease c by threshold
						this.decreaseComputationTimeByThreshold(selectedFilter,
								c, c_value, c_threshold, tenant);
					} else {
						// decreased s
						LOGGER.info(" -- Selection rate of filter \""
								+ selectedFilter.getUid() + "\" in tenant \""
								+ tenant.getName()
								+ "\" has been decreased from " + s_value
								+ " to " + new_s_value);

						// increase c
						this.increaseComputationTimeByThreshold(selectedFilter,
								c, c_value, c_threshold, tenant);
					}

				}

			}
			return true;

	}

	private void increaseSelectionRateByThreshold(Component filter,
			MonitoredProperty s, double current_s_value, double s_threshold,
			Tenant tenant) {
		double new_s_value = current_s_value * (1 + s_threshold);
		s.setValue("" + new_s_value);
		LOGGER.info(" -- Selection rate of filter \"" + filter.getUid()
				+ "\" in tenant \"" + tenant.getName()
				+ "\" has been increased from " + current_s_value + " to "
				+ new_s_value);
	}

	private void decreaseSelectionRateByThreshold(Component filter,
			MonitoredProperty s, double current_s_value, double s_threshold,
			Tenant tenant) {
		double new_s_value = current_s_value * (1 - s_threshold);
		s.setValue("" + new_s_value);
		LOGGER.info(" -- Selection rate of filter \"" + filter.getUid()
				+ "\" in tenant \"" + tenant.getName()
				+ "\" has been decreased from " + current_s_value + " to "
				+ new_s_value);
	}

	private void increaseComputationTimeByThreshold(Component filter,
			MonitoredProperty c, long current_c_value, double c_threshold,
			Tenant tenant) {
		long new_c_value = (long) (current_c_value * (1 + c_threshold));
		c.setValue("" + new_c_value);
		LOGGER.info(" -- Local computation time of filter \"" + filter.getUid()
				+ "\" in tenant \"" + tenant.getName()
				+ "\" has been increased from " + current_c_value + " to "
				+ new_c_value);
	}

	private void decreaseComputationTimeByThreshold(Component filter,
			MonitoredProperty c, long current_c_value, double c_threshold,
			Tenant tenant) {
		long new_c_value = (long) (current_c_value * (1 - c_threshold));
		c.setValue("" + new_c_value);
		LOGGER.info(" -- Local computation time of filter \"" + filter.getUid()
				+ "\" in tenant \"" + tenant.getName()
				+ "\" has been decreased from " + current_c_value + " to "
				+ new_c_value);
	}

	private double calculateScalarForTheSelectionRate(List<Component> filters,
			Component selectedFilter, long current_c_value,
			double current_s_value) {

		double slopeOfSelectedFilter = current_s_value / current_c_value;
		int positionOfSelectedFilter = filters.indexOf(selectedFilter);

		double scalar = 1;

		if (filters.size() >= 4) {
			int halfThePipe = filters.size() / 2;

			// the selected filter should be shifted by one or two positions
			int positionChange = SimulatorUtil.randInt(1, 2);
			// ... where the other filter is located
			Component otherFilter = null;
			if (positionOfSelectedFilter >= halfThePipe) {
				// selected filter is located in the second half of the pipe.
				// it should be placed by an adaptation toward the beginning of
				// the pipe. hence, selection rate should be increased.
				if (positionOfSelectedFilter - positionChange >= 0) {
					otherFilter = filters.get(positionOfSelectedFilter
							- positionChange);
				}
			} else {
				// selected filter is located in the first half of the pipe.
				// it should be placed by an adaptation toward the end of the
				// pipe. hence, selection rate should be decreased.
				if (positionOfSelectedFilter + positionChange < filters.size()) {
					otherFilter = filters.get(positionOfSelectedFilter
							+ positionChange);
				}
			}

			double slopeOfOtherFilter = MrubisModelQuery.INSTANCE
					.getSlopeOfFilter(otherFilter);
			scalar = (slopeOfOtherFilter / slopeOfSelectedFilter);

		} else if (filters.size() == 3) {
			// there are 3 filters in the pipe.
			// switch with neighbor filter
			Component neighborFilter = null;
			int deltaPosition = 0;
			if (positionOfSelectedFilter == 0) {
				deltaPosition = 1;
			} else if (positionOfSelectedFilter == 1) {
				int leftOrRightNeighbor = SimulatorUtil.randInt(0, 1);
				if (leftOrRightNeighbor == 0) {
					deltaPosition = -1;
				} else {
					deltaPosition = 1;
				}
			} else if (positionOfSelectedFilter == 2) {
				deltaPosition = -1;
			}
			neighborFilter = filters.get(positionOfSelectedFilter
					+ deltaPosition);
			double slopeOfOtherFilter = MrubisModelQuery.INSTANCE
					.getSlopeOfFilter(neighborFilter);
			scalar = (slopeOfOtherFilter / slopeOfSelectedFilter);

		} else if (filters.size() == 2) {
			// there are 2 filters in the pipe.
			// switch with neighbor
			Component neighborFilter = null;
			if (positionOfSelectedFilter == 0) {
				neighborFilter = filters.get(1);
			} else if (positionOfSelectedFilter == 1) {
				neighborFilter = filters.get(0);
			}
			double slopeOfOtherFilter = MrubisModelQuery.INSTANCE
					.getSlopeOfFilter(neighborFilter);
			scalar = (slopeOfOtherFilter / slopeOfSelectedFilter);
		} else {
			LOGGER.warning(" -- [INFO] There is only 1 filter in the pipe.");
		}

		return scalar;
	}

}
