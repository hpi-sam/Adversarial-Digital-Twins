package de.mdelab.morisia.comparch.simulator.mrubis.analyzer.adaptation;

import java.util.List;
import java.util.logging.Logger;

import org.eclipse.emf.ecore.util.EcoreUtil;

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.ComparchFactory;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Connector;
import de.mdelab.morisia.comparch.MonitoredProperty;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.RequiredInterface;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.comparch.simulator.AdaptationAnalyzer;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisModelQuery;
import de.mdelab.morisia.comparch.simulator.mrubis.MrubisNames;

/**
 * Analyzes an adaptation that deploys a new filter. For instance, if a filter
 * component has been removed, and a new filter component of the same type has
 * been deployed, this new filter component has no {@link MonitoredProperty}
 * elements. They should be added, which is done by this analyzer.
 * 
 * Usually, the monitored properties are added to the model by monitoring the
 * running system.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public class MissingMonitoredPropertyAnalyzer implements AdaptationAnalyzer {

	private final static Logger LOGGER = Logger
			.getLogger(MissingMonitoredPropertyAnalyzer.class.getName());

	/**
	 * Constructor.
	 */
	public MissingMonitoredPropertyAnalyzer() {

	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getName() {
		return "Missing Monitored Property Analyzer";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public String getDescription() {
		return "Analyzes the monitored properties of filter components and especially adds "
				+ "missing properties of filters that have been newly deployed by an adaptation.";
	}

	/**
	 * {@inheritDoc}
	 */
	@Override
	public int analyze(Architecture eArchitecture) {

		for (Tenant tenant : eArchitecture.getTenants()) {
			LOGGER.info("Analyzing the tenant \"" + tenant.getName() + "\" ...");

			List<Component> filters = MrubisModelQuery.INSTANCE
					.getAllFiltersOfTenant(tenant);

			boolean missingElements = false;
			boolean blackListTenant = false;
			int position = 0;
			for (Component filter : filters) {
				// MonitoredProperty: selection rate
				MonitoredProperty selectionRate = MrubisModelQuery.INSTANCE
						.getMonitoredPropertyByName(filter,
								MrubisNames.SELECTION_RATE_PROPERTY_NAME);

				// Property does not exist. Create it with an empty value.
				if (selectionRate == null) {
					selectionRate = this.createSelectionRateProperty(null);
					filter.getMonitoredProperties().add(selectionRate);
					LOGGER.finer(" -- Filter \""
							+ filter.getUid()
							+ "\" of tenant \""
							+ tenant.getName()
							+ "\" has no MonitoredProperty for the selection rate. Property without a concrete value has been added.");
				}

				// Property exists but has no value. Calculate and set the
				// value.
				String value = selectionRate.getValue();

				LOGGER.finer(" -- Filter \""
						+ filter.getUid()
						+ "\" of tenant \""
						+ tenant.getName()
						+ "\" has a MonitoredProperty with a selection rate of "
						+ value);

				if (value == null || value.equals("")) {
					// Selection rate to be calculated.
					double selectionRateValue = 0;
					if (position == 0) {
						// first filter
						if (filters.size() >= 2) {
							Component rightNeighbor = filters.get(1);
							double neighborSelectionRateValue = this
									.getSelectionRateOfFilter(rightNeighbor);
							selectionRateValue = neighborSelectionRateValue * 1.1;
						} else {
							// there is only one filter in the pipe. Assign a
							// selection rate of 0.25
							selectionRateValue = 0.25;
						}

					} else if (position >= 1 && position < filters.size() - 1) {
						// filter in between
						Component leftNeighbor = filters.get(position - 1);
						double leftNeighborSelectionRateValue = this
								.getSelectionRateOfFilter(leftNeighbor);
						Component rightNeighbor = filters.get(position + 1);
						double rightNeighborSelectionRateValue = this
								.getSelectionRateOfFilter(rightNeighbor);
						selectionRateValue = (leftNeighborSelectionRateValue + rightNeighborSelectionRateValue) / 2;

					} else if (position > 0 && position == filters.size() - 1) {
						// last filter
						// there is at least one filter before
						Component leftNeighbor = filters.get(position - 1);
						double leftNeighborSelectionRateValue = this
								.getSelectionRateOfFilter(leftNeighbor);
						selectionRateValue = leftNeighborSelectionRateValue * 0.9;

					}

					// set the selection rate value
					selectionRate.setValue("" + selectionRateValue);

					blackListTenant = true;
					missingElements = true;
					LOGGER.info("  -- Filter \"" + filter.getUid()
							+ "\" of tenant \"" + tenant.getName()
							+ "\" has been set the selection rate of "
							+ selectionRateValue);

				}

				// MonitoredProperty: local computation time
				MonitoredProperty localComputationTime = MrubisModelQuery.INSTANCE
						.getMonitoredPropertyByName(
								filter,
								MrubisNames.LOCAL_COMPUTATION_TIME_PROPERTY_NAME);

				// Property does not exist. Create it with an empty value.
				if (localComputationTime == null) {
					localComputationTime = this
							.createLocalComputationTimeProperty(null);
					filter.getMonitoredProperties().add(localComputationTime);
					LOGGER.finer(" -- Filter \""
							+ filter.getUid()
							+ "\" of tenant \""
							+ tenant.getName()
							+ "\" has no MonitoredProperty for the local computation time. Property without a concrete value has been added.");
				}

				// Property exists but has no value. Calculate and set the
				// value.
				String time = localComputationTime.getValue();

				LOGGER.finer(" -- Filter \""
						+ filter.getUid()
						+ "\" of tenant \""
						+ tenant.getName()
						+ "\" has a MonitoredProperty with a local computation time of "
						+ time);

				if (time == null || time.equals("")) {
					// time to be calculated
					long compTimeValue = 0;
					// add property
					if (position == 0) {
						// first filter
						if (filters.size() >= 2) {
							Component rightNeighbor = filters.get(1);
							long neighborCompTimeValue = this
									.getComputationTimeOfFilter(rightNeighbor);
							compTimeValue = (long) (neighborCompTimeValue * 0.95);
						} else {
							// there is only one filter in the pipe. Assign a
							// computation time of 45
							compTimeValue = 45;
						}

					} else if (position >= 1 && position < filters.size() - 1) {
						// filter in between
						Component leftNeighbor = filters.get(position - 1);
						long leftNeighborCompTimeValue = this
								.getComputationTimeOfFilter(leftNeighbor);
						Component rightNeighbor = filters.get(position + 1);
						long rightNeighborCompTimeValue = this
								.getComputationTimeOfFilter(rightNeighbor);
						compTimeValue = (leftNeighborCompTimeValue + rightNeighborCompTimeValue) / 2;

					} else if (position > 0 && position == filters.size() - 1) {
						// last filter
						// there is at least one filter before
						Component leftNeighbor = filters.get(position - 1);
						long leftNeighborCompTimeValue = this
								.getComputationTimeOfFilter(leftNeighbor);
						compTimeValue = (long) (leftNeighborCompTimeValue * 1.05);

					}

					// Set the time value to the property.
					localComputationTime.setValue("" + compTimeValue);

					blackListTenant = true;
					missingElements = true;
					LOGGER.info("  -- Filter \"" + filter.getUid()
							+ "\" of tenant \"" + tenant.getName()
							+ "\" has been set the local computation time of "
							+ compTimeValue);

				}

				// =========================
				position = position + 1;
			} // next filter

			if (blackListTenant) {
				// TODO blacklist tenant?
			}

			if (!missingElements && filters.size() > 0) {
				LOGGER.info("  -- All filters of tenant \""
						+ tenant.getName()
						+ "\" have the MonitoredPropery elements for the selection rate and local computation time.\n");
			} else if (missingElements && filters.size() > 0) {
				LOGGER.info("  -- All filters of tenant \""
						+ tenant.getName()
						+ "\" now have the MonitoredPropery elements for the selection rate and local computation time.\n");
			}

		} // next tenant
		return 0;
	}

	private MonitoredProperty createSelectionRateProperty(String value) {
		MonitoredProperty eMonitoredProperty = ComparchFactory.eINSTANCE
				.createMonitoredProperty();
		eMonitoredProperty.setUid(EcoreUtil.generateUUID());
		eMonitoredProperty.setName(MrubisNames.SELECTION_RATE_PROPERTY_NAME);
		if (value != null) {
			eMonitoredProperty.setValue(value);
		}
		eMonitoredProperty.setType(MrubisNames.SELECTION_RATE_PROPERTY_TYPE);
		eMonitoredProperty.setUnit(MrubisNames.SELECTION_RATE_PROPERTY_UNIT);
		eMonitoredProperty
				.setDescription(MrubisNames.SELECTION_RATE_PROPERTY_DESCRIPTION);
		return eMonitoredProperty;
	}

	private MonitoredProperty createLocalComputationTimeProperty(String value) {
		MonitoredProperty eMonitoredProperty = ComparchFactory.eINSTANCE
				.createMonitoredProperty();
		eMonitoredProperty.setUid(EcoreUtil.generateUUID());
		eMonitoredProperty
				.setName(MrubisNames.LOCAL_COMPUTATION_TIME_PROPERTY_NAME);
		if (value != null) {
			eMonitoredProperty.setValue(value);
		}
		eMonitoredProperty
				.setType(MrubisNames.LOCAL_COMPUTATION_TIME_PROPERTY_TYPE);
		eMonitoredProperty
				.setUnit(MrubisNames.LOCAL_COMPUTATION_TIME_PROPERTY_UNIT);
		eMonitoredProperty
				.setDescription(MrubisNames.LOCAL_COMPUTATION_TIME_PROPERTY_DESCRIPTION);
		return eMonitoredProperty;
	}

	private long getComputationTimeOfFilter(Component filter) {
		MonitoredProperty c = MrubisModelQuery.INSTANCE
				.getMonitoredPropertyByName(filter,
						MrubisNames.LOCAL_COMPUTATION_TIME_PROPERTY_NAME);
		if (c != null) {
			return Long.parseLong(c.getValue());
		} else {
			String msg = "Local computation time property of filter \""
					+ filter.getUid() + "\" is missing => Exception :(";
			LOGGER.severe(msg);
			this.debug(filter);
			throw new RuntimeException(msg);
		}
	}

	private double getSelectionRateOfFilter(Component filter) {
		MonitoredProperty c = MrubisModelQuery.INSTANCE
				.getMonitoredPropertyByName(filter,
						MrubisNames.SELECTION_RATE_PROPERTY_NAME);
		if (c != null) {
			return Double.parseDouble(c.getValue());
		} else {
			String msg = "Selection rate property of filter \""
					+ filter.getUid() + "\" is missing => Exception :(";
			LOGGER.severe(msg);
			this.debug(filter);
			throw new RuntimeException(msg);
		}
	}

	private void debug(Component filter) {
		String tenantName = null;
		if (filter.getTenant() != null) {
			tenantName = filter.getTenant().getName();
		}

		String msg = "How is the filter \"" + filter.getUid()
				+ "\" of tenant \"" + tenantName + "\" connected?\n";
		msg = msg + " -- It is used by:\n";
		// Connected by
		for (ProvidedInterface pi : filter.getProvidedInterfaces()) {
			for (Connector inCon : pi.getConnectors()) {
				RequiredInterface sourceInCon = inCon.getSource();
				if (sourceInCon != null) {
					msg = msg + "    -- Component \""
							+ sourceInCon.getComponent().getUid() + "\"\n";
				} else {
					msg = msg + "    -- no component.\n";
				}
			}
			if (pi.getConnectors().size() == 0) {
				msg = msg + "    -- no component.\n";
			}
		}
		// Connected to
		msg = msg + " -- It is using:\n";
		for (RequiredInterface ri : filter.getRequiredInterfaces()) {
			Connector outCon = ri.getConnector();
			if (outCon != null && outCon.getTarget() != null) {
				msg = msg + "    -- Component \""
						+ outCon.getTarget().getComponent().getUid() + "\"\n";
			} else {
				msg = msg + "    -- no component.\n";
			}
		}
		LOGGER.finer(msg);
	}

}
