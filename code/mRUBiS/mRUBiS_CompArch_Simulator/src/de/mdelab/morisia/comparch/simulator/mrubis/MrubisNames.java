package de.mdelab.morisia.comparch.simulator.mrubis;

/**
 * Constants and names used in the mRUBiS/CompArch model.
 * 
 * @author thomas vogel
 * @version 0.21
 *
 */
public interface MrubisNames {

	public final static String BASE_PACKAGE = "de.hpi.sam.rubis.";
	public final static String ABOUT_ME_SERIVICE_INTERFACE = BASE_PACKAGE
			+ "usermgmt.AboutMeService";
	public final static String VIEW_USER_INFO_SERVICE_INTERFACE = BASE_PACKAGE
			+ "usermgmt.ViewUserInfoService";
	public final static String BROWSE_REGION_SERVICE_INTERFACE = BASE_PACKAGE
			+ "usermgmt.BrowseRegionsService";
	public final static String USER_REGISTRATION_SERVICE_INTERFACE = BASE_PACKAGE
			+ "usermgmt.UserRegistrationService";
	public final static String BROWSE_CATEGORIES_SERVICE_INTERFACE = BASE_PACKAGE
			+ "itemmgmt.BrowseCategoriesService";
	public final static String ITEM_REGISTRATION_SERVICE_INTERFACE = BASE_PACKAGE
			+ "itemmgmt.ItemRegistrationService";
	public final static String AUTHENTICATION_SERVICE_INTERFACE = BASE_PACKAGE
			+ "authservice.AuthenticationService";
	public final static String BID_SERVICE_INTERFACE = BASE_PACKAGE
			+ "bidandbuy.BidService";
	public final static String BUY_NOW_SERVICE_INTERFACE = BASE_PACKAGE
			+ "bidandbuy.BuyNowService";
	public final static String INVENTORY_SERVICE_INTERFACE = BASE_PACKAGE
			+ "inventorymgmt.InventoryService";
	public final static String REPUTATION_SERVICE_INTERFACE = BASE_PACKAGE
			+ "reputationservice.ReputationService";
	public final static String BO_PERSISTENCE_SERVICE_INTERFACE = BASE_PACKAGE
			+ "persistenceservice.BusinessObjectsPersistenceService";
	public final static String BASIC_QUERY_SERVICE_INTERFACE = BASE_PACKAGE
			+ "queryservice.BasicQueryService";
	public final static String QUERY_SERVICE_INTERFACE = BASE_PACKAGE
			+ "queryservice.QueryService";
	public final static String ITEM_FILTER_SERVICE_INTERFACE = BASE_PACKAGE
			+ "filter.ItemFilter";

	public final static String SELECTION_RATE_PROPERTY_NAME = "selection-rate";
	public final static String SELECTION_RATE_PROPERTY_TYPE = Double.class
			.getCanonicalName();
	public final static String SELECTION_RATE_PROPERTY_UNIT = "*100%";
	public final static String SELECTION_RATE_PROPERTY_DESCRIPTION = "The selection rate s of this filter";

	public final static String LOCAL_COMPUTATION_TIME_PROPERTY_NAME = "local-computation-time";
	public final static String LOCAL_COMPUTATION_TIME_PROPERTY_TYPE = Long.class
			.getCanonicalName();
	public final static String LOCAL_COMPUTATION_TIME_PROPERTY_UNIT = "ms";
	public final static String LOCAL_COMPUTATION_TIME_PROPERTY_DESCRIPTION = "The local computation time c of this filter";

	public final static String SELECTION_RATE_THRESHOLD_PARAMETER_NAME = "selection-rate-threshold";
	public final static String LOCAL_COMPUTATION_TIME_THRESHOLD_PARAMETER_NAME = "computation-time-threshold";

	public final static String GET_PERSONALIZED_ITEMS_SIGNATURE = "getPersonalizedItems(java.lang.String,java.lang.String)";
}
