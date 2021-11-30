package de.mdelab.morisia.mrubis;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;

/**
 * Generates performance data of mRUBiS.
 * 
 * @author thomas
 * 
 */
public class CompArchPerformanceDataGenerator {

	public static double SCALE = 1.00;

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		CompArchPerformanceDataGenerator gen = new CompArchPerformanceDataGenerator();
		gen.generatePerformanceStats();
		gen.generateMonitoredProperties();
	}

	// String is the component (type) name
	Map<String, Map<Method, MethodPerformance>> generatePerformanceStats() {
		Map<String, Map<Method, MethodPerformance>> data = new HashMap<>();

		Method filterMethod = new Method("de.hpi.sam.rubis.filter.ItemFilter",
				"filter(java.util.List,de.hpi.sam.rubis.entity.User)");
		// Future Sales Item Filter
		Map<Method, MethodPerformance> futureSalesIfPerf = new HashMap<>();
		data.put(MRubisNames.futureSalesItemFilter_ComponentType_Name,
				futureSalesIfPerf);
		futureSalesIfPerf.put(filterMethod, new MethodPerformance(15, 100,
				30 * 30, 30));
		// Recommendation Item Filter
		Map<Method, MethodPerformance> recIfPerf = new HashMap<>();
		data.put(MRubisNames.recommendationItemFilter_ComponentType_Name,
				recIfPerf);
		recIfPerf
				.put(filterMethod, new MethodPerformance(45, 175, 67 * 30, 30));
		// Category Item Filter
		Map<Method, MethodPerformance> categoryIfPerf = new HashMap<>();
		data.put(MRubisNames.categoryItemFilter_ComponentType_Name,
				categoryIfPerf);
		categoryIfPerf.put(filterMethod, new MethodPerformance(95, 300,
				136 * 30, 30));
		// Comment Item Filter
		Map<Method, MethodPerformance> commentIfPerf = new HashMap<>();
		data.put(MRubisNames.commentItemFilter_ComponentType_Name,
				commentIfPerf);
		commentIfPerf.put(filterMethod, new MethodPerformance(140, 390,
				204 * 30, 30));
		// Seller Reputation Item Filter
		Map<Method, MethodPerformance> sellerReputationIfPerf = new HashMap<>();
		data.put(MRubisNames.sellerReputationItemFilter_ComponentType_Name,
				sellerReputationIfPerf);
		sellerReputationIfPerf.put(filterMethod, new MethodPerformance(190,
				440, 269 * 30, 30));
		// Region item filter
		Map<Method, MethodPerformance> regionIfPerf = new HashMap<>();
		data.put(MRubisNames.regionItemFilter_ComponentType_Name, regionIfPerf);
		regionIfPerf.put(filterMethod, new MethodPerformance(240, 500,
				332 * 30, 30));
		// Availability Item Filter
		Map<Method, MethodPerformance> availabiliyIfPerf = new HashMap<>();
		data.put(MRubisNames.availabilityItemFilter_ComponentType_Name,
				availabiliyIfPerf);
		availabiliyIfPerf.put(filterMethod, new MethodPerformance(300, 620,
				392 * 30, 30));
		// Buy-Now Item Filter
		Map<Method, MethodPerformance> buynowIfPerf = new HashMap<>();
		data.put(MRubisNames.buyNowItemFilter_ComponentType_Name, buynowIfPerf);
		buynowIfPerf.put(filterMethod, new MethodPerformance(340, 690,
				449 * 30, 30));
		// Past Sales Item Filter
		Map<Method, MethodPerformance> pastSalesIfPerf = new HashMap<>();
		data.put(MRubisNames.pastSalesItemFilter_ComponentType_Name,
				pastSalesIfPerf);
		pastSalesIfPerf.put(filterMethod, new MethodPerformance(400, 740,
				503 * 30, 30));
		// Last Second Sales Item Filter
		Map<Method, MethodPerformance> lastSecondSalesIfPerf = new HashMap<>();
		data.put(MRubisNames.lastSecondSalesItemFilter_ComponentType_Name,
				lastSecondSalesIfPerf);
		lastSecondSalesIfPerf.put(filterMethod, new MethodPerformance(450, 760,
				560 * 30, 30));

		// Query Service
		Map<Method, MethodPerformance> queryServicePerf = new HashMap<>();
		data.put(MRubisNames.queryService_ComponentType_Name, queryServicePerf);
		String basicQueryServiceInterface = "de.hpi.sam.rubis.queryservice.BasicQueryService";
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findAllUsers()"), new MethodPerformance(0, 0, 0, 0));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findUserById(int)"), new MethodPerformance(0, 377, 896, 9));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findUserByNickname(java.lang.String)"), new MethodPerformance(
				2, 479, 3495, 48));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findAllCategories()"),
				new MethodPerformance(184, 619, 1134, 3));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findCategoryById(int)"), new MethodPerformance(167, 717, 1210,
				3));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findCategoryByName(java.lang.String)"), new MethodPerformance(
				0, 0, 0, 0));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findCategoriesByName(java.lang.String)"),
				new MethodPerformance(0, 0, 0, 0));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findAllRegions()"), new MethodPerformance(194, 1484, 3335, 6));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findRegionById(int)"), new MethodPerformance(0, 0, 0, 0));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findRegionByName(java.lang.String)"), new MethodPerformance(5,
				442, 977, 6));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findRegionsByName(java.lang.String)"), new MethodPerformance(
				0, 0, 0, 0));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findItemsByName(java.lang.String)"), new MethodPerformance(
				188, 375, 819, 3));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findItemById(int)"), new MethodPerformance(0, 1, 2, 9));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findAllItems()"), new MethodPerformance(0, 0, 0, 0));
		/*
		 * findPersonalizedItems(de.hpi.sam.rubis.entity.User)
		 */
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"findPersonalizedItems(de.hpi.sam.rubis.entity.User)"),
				new MethodPerformance(600, 900, 816 * 30, 30));
		queryServicePerf.put(new Method(basicQueryServiceInterface,
				"isUserNicknameUsed(java.lang.String)"), new MethodPerformance(
				4, 12, 43, 6));

		String queryServiceInterface = "de.hpi.sam.rubis.queryservice.QueryService";
		queryServicePerf.put(new Method(queryServiceInterface,
				"findCurrentItemsInCategory(int,int)"), new MethodPerformance(
				0, 0, 0, 0));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findPastItemsInCategory(int,int)"), new MethodPerformance(0,
				0, 0, 0));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findCurrentItemsInCategoryAndRegion(int,int,int)"),
				new MethodPerformance(0, 0, 0, 0));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findPastItemsInCategoryAndRegion(int,int,int)"),
				new MethodPerformance(0, 0, 0, 0));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findItemMaxBid(int)"), new MethodPerformance(0, 0, 0, 0));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findItemMaxBids(int,int)"), new MethodPerformance(11, 393,
				881, 6));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findItemBidHistory(int)"), new MethodPerformance(0, 0, 0, 0));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findSuccessfulBidsForItem(int)"), new MethodPerformance(0, 0,
				0, 0));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findUserBuyNows(int,int)"), new MethodPerformance(0, 0, 0, 0));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findUserCurrentSellings(int)"), new MethodPerformance(0, 0, 0,
				0));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findUserPastSellings(int,int)"), new MethodPerformance(0, 0,
				0, 0));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findUserPastWinningBids(int)"), new MethodPerformance(12, 38,
				118, 6));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findUserCurrentMaxBids(int)"), new MethodPerformance(12, 20,
				91, 6));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findCategoriesInRegion(java.lang.String)"),
				new MethodPerformance(0, 0, 0, 0));
		queryServicePerf.put(new Method(queryServiceInterface,
				"getNumberOfBidsForItem(int)"), new MethodPerformance(0, 0, 0,
				0));
		queryServicePerf.put(new Method(queryServiceInterface,
				"findComment(int,int,int)"), new MethodPerformance(42, 93, 199,
				3));
		queryServicePerf.put(new Method(queryServiceInterface,
				"retrieveAvailabilityOfItem(int)"), new MethodPerformance(4,
				11, 64, 9));

		// Persistence Service
		Map<Method, MethodPerformance> persistenceServicePerf = new HashMap<>();
		data.put(MRubisNames.persistenceService_ComponentType_Name,
				persistenceServicePerf);
		String persistenceServiceInterface = "de.hpi.sam.rubis.persistenceservice.BusinessObjectsPersistenceService";
		persistenceServicePerf
				.put(new Method(
						persistenceServiceInterface,
						"persistUser(java.lang.String,java.lang.String,java.lang.String,java.lang.String,java.lang.String,de.hpi.sam.rubis.entity.Region,de.hpi.sam.rubis.entity.CustomerClass)"),
						new MethodPerformance(1, 433, 969, 6));
		persistenceServicePerf
				.put(new Method(
						persistenceServiceInterface,
						"persistItem(java.lang.String,java.lang.String,int,float,float,float,java.util.Date,java.util.Date,de.hpi.sam.rubis.entity.User,de.hpi.sam.rubis.entity.Category)"),
						new MethodPerformance(170, 550, 1063, 3));
		persistenceServicePerf
				.put(new Method(
						persistenceServiceInterface,
						"persistBid(de.hpi.sam.rubis.entity.User,de.hpi.sam.rubis.entity.Item,float,float,int,java.util.Date)"),
						new MethodPerformance(200, 445, 935, 3));
		persistenceServicePerf
				.put(new Method(
						persistenceServiceInterface,
						"persistBuyNow(de.hpi.sam.rubis.entity.User,de.hpi.sam.rubis.entity.Item,int,java.util.Date)"),
						new MethodPerformance(61, 122, 301, 3));
		persistenceServicePerf
				.put(new Method(
						persistenceServiceInterface,
						"persistComment(de.hpi.sam.rubis.entity.User,de.hpi.sam.rubis.entity.User,de.hpi.sam.rubis.entity.Item,java.lang.String,int,java.util.Date)"),
						new MethodPerformance(169, 353, 737, 3));
		persistenceServicePerf.put(new Method(persistenceServiceInterface,
				"reduceInventoryItem(de.hpi.sam.rubis.entity.Item,int)"),
				new MethodPerformance(10, 15, 39, 3));
		persistenceServicePerf.put(new Method(persistenceServiceInterface,
				"increaseInventoryItem(de.hpi.sam.rubis.entity.Item,int)"),
				new MethodPerformance(0, 0, 0, 0));

		// Reputation Service
		Map<Method, MethodPerformance> reputationServicePerf = new HashMap<>();
		data.put(MRubisNames.reputationService_ComponentType_Name,
				reputationServicePerf);
		reputationServicePerf
				.put(new Method(
						"de.hpi.sam.rubis.reputationservice.ReputationService",
						"giveComment(java.lang.String,java.lang.String,int,int,java.lang.String,int)"),
						new MethodPerformance(436, 793, 1770, 3));

		// Inventory Service
		Map<Method, MethodPerformance> inventoryServicePerf = new HashMap<>();
		data.put(MRubisNames.inventoryService_ComponentType_Name,
				inventoryServicePerf);
		String inventoryServiceInterface = "de.hpi.sam.rubis.inventorymgmt.InventoryService";
		inventoryServicePerf.put(new Method(inventoryServiceInterface,
				"checkAvailabilityOfItem(de.hpi.sam.rubis.entity.Item)"),
				new MethodPerformance(7, 37, 109, 6));
		inventoryServicePerf.put(new Method(inventoryServiceInterface,
				"reserveItem(de.hpi.sam.rubis.entity.Item,int)"),
				new MethodPerformance(30, 40, 103, 3));
		inventoryServicePerf.put(new Method(inventoryServiceInterface,
				"cancelReservedItem(de.hpi.sam.rubis.entity.Item,int)"),
				new MethodPerformance(0, 0, 0, 0));

		// Bid and Buy Service
		Map<Method, MethodPerformance> bidandbuyServicePerf = new HashMap<>();
		data.put(MRubisNames.bidAndBuyService_ComponentType_Name,
				bidandbuyServicePerf);
		String bidServiceInterface = "de.hpi.sam.rubis.bidandbuy.BidService";
		bidandbuyServicePerf
				.put(new Method(bidServiceInterface,
						"bidOnItem(int,float,float,int,java.lang.String,java.lang.String)"),
						new MethodPerformance(423, 1086, 2138, 3));
		bidandbuyServicePerf.put(new Method(bidServiceInterface,
				"getItemBidHistory(int)"), new MethodPerformance(0, 0, 0, 0));
		bidandbuyServicePerf.put(new Method(bidServiceInterface,
				"getItemBidHistorySortyByBidPrice(int)"),
				new MethodPerformance(13, 402, 908, 6));

		bidandbuyServicePerf.put(new Method(
				"de.hpi.sam.rubis.bidandbuy.BuyNowService",
				"buyItemNow(int,int,java.lang.String,java.lang.String)"),
				new MethodPerformance(335, 620, 1375, 3));

		// Authentication Service
		Map<Method, MethodPerformance> authenticationServicePerf = new HashMap<>();
		data.put(MRubisNames.authenticationService_ComponentType_Name,
				authenticationServicePerf);
		/*
		 * authenticate(java.lang.String,java.lang.String)
		 */
		authenticationServicePerf.put(new Method(
				"de.hpi.sam.rubis.authservice.AuthenticationService",
				"authenticate(java.lang.String,java.lang.String)"),
				new MethodPerformance(50, 153, 3675, 48));

		// Item Management Service
		Map<Method, MethodPerformance> itemMgmtServicePerf = new HashMap<>();
		data.put(MRubisNames.itemManagementService_ComponentType_Name,
				itemMgmtServicePerf);
		itemMgmtServicePerf
				.put(new Method(
						"de.hpi.sam.rubis.itemmgmt.ItemRegistrationService",
						"registerItem(java.lang.String,java.lang.String,int,float,float,float,java.util.Date,java.util.Date,int,java.lang.String,int)"),
						new MethodPerformance(363, 1349, 2425, 3));
		String browseCategoriesServiceInterface = "de.hpi.sam.rubis.itemmgmt.BrowseCategoriesService";
		itemMgmtServicePerf.put(new Method(browseCategoriesServiceInterface,
				"getCategoriesInRegion(java.lang.String)"),
				new MethodPerformance(0, 0, 0, 0));
		itemMgmtServicePerf
				.put(new Method(browseCategoriesServiceInterface,
						"getAllCategories()"), new MethodPerformance(190, 652,
						1185, 3));
		itemMgmtServicePerf.put(new Method(browseCategoriesServiceInterface,
				"getCategoriesByName(java.lang.String)"),
				new MethodPerformance(0, 0, 0, 0));
		itemMgmtServicePerf.put(new Method(browseCategoriesServiceInterface,
				"getItemsByName(java.lang.String)"), new MethodPerformance(196,
				391, 853, 3));
		/*
		 * getPersonalizedItems(java.lang.String,java.lang.String)
		 */
		itemMgmtServicePerf.put(new Method(browseCategoriesServiceInterface,
				"getPersonalizedItems(java.lang.String,java.lang.String)"),
				new MethodPerformance(680, 996, 897 * 30, 30));

		// User Management Service
		Map<Method, MethodPerformance> userMgmtServicePerf = new HashMap<>();
		data.put(MRubisNames.userManagementService_ComponentType_Name,
				userMgmtServicePerf);
		userMgmtServicePerf.put(new Method(
				"de.hpi.sam.rubis.usermgmt.AboutMeService",
				"getAboutMe(java.lang.String,java.lang.String)"),
				new MethodPerformance(39, 556, 1294, 6));
		userMgmtServicePerf.put(new Method(
				"de.hpi.sam.rubis.usermgmt.ViewUserInfoService",
				"getUserInfo(int)"), new MethodPerformance(188, 404, 907, 3));
		userMgmtServicePerf.put(new Method(
				"de.hpi.sam.rubis.usermgmt.BrowseRegionsService",
				"getAllRegions()"), new MethodPerformance(203, 1550, 3459, 6));
		userMgmtServicePerf
				.put(new Method(
						"de.hpi.sam.rubis.usermgmt.UserRegistrationService",
						"registerUser(java.lang.String,java.lang.String,java.lang.String,java.lang.String,java.lang.String,java.lang.String,de.hpi.sam.rubis.entity.CustomerClass)"),
						new MethodPerformance(13, 937, 2109, 6));

		return data;

	}

	Map<String, List<Property>> generateMonitoredProperties() {
		Map<String, List<Property>> props = new HashMap<>();

		// createProperties(String selectionRateValue, String
		// localComputationTimeValue)

		// Future Sales Item Filter
		props.put(MRubisNames.futureSalesItemFilter_ComponentType_Name,
				this.createProperties("0.03", "30"));

		// Recommendation Item Filter
		props.put(MRubisNames.recommendationItemFilter_ComponentType_Name,
				this.createProperties("0.04", "37"));

		// Category Item Filter
		props.put(MRubisNames.categoryItemFilter_ComponentType_Name,
				this.createProperties("0.08", "69"));

		// Comment Item Filter
		props.put(MRubisNames.commentItemFilter_ComponentType_Name,
				this.createProperties("0.10", "68"));

		// Seller Reputation Item Filter
		props.put(MRubisNames.sellerReputationItemFilter_ComponentType_Name,
				this.createProperties("0.12", "65"));

		// Region item filter
		props.put(MRubisNames.regionItemFilter_ComponentType_Name,
				this.createProperties("0.15", "63"));

		// Availability Item Filter
		props.put(MRubisNames.availabilityItemFilter_ComponentType_Name,
				this.createProperties("0.17", "60"));

		// Buy-Now Item Filter
		props.put(MRubisNames.buyNowItemFilter_ComponentType_Name,
				this.createProperties("0.18", "57"));

		// Past Sales Item Filter
		props.put(MRubisNames.pastSalesItemFilter_ComponentType_Name,
				this.createProperties("0.20", "54"));

		// Last Second Sales Item Filter
		props.put(MRubisNames.lastSecondSalesItemFilter_ComponentType_Name,
				this.createProperties("0.22", "57"));

		return props;

	}

	private List<Property> createProperties(String selectionRateValue,
			String localComputationTimeValue) {
		List<Property> props = new LinkedList<>();
		props.add(this.createSelectionRateProperty(selectionRateValue));
		props.add(this
				.createLocalComputationTimeProperty(localComputationTimeValue));
		return props;
	}

	private String selectionRateName = "selection-rate";
	private String selectionRateType = Double.class.getCanonicalName();
	private String selectionRateUnit = "*100%";
	private String selectionRateDescription = "The selection rate s of this filter";

	private String localComputationTimeName = "local-computation-time";
	private String localComputationTimeType = Long.class.getCanonicalName();
	private String localComputationTimeUnit = "ms";
	private String localComputationTimeDescription = "The local computation time c of this filter";

	private Property createSelectionRateProperty(String value) {
		return new Property(selectionRateName, value, selectionRateType,
				selectionRateUnit, selectionRateDescription);
	}

	private Property createLocalComputationTimeProperty(String value) {
		return new Property(localComputationTimeName, value,
				localComputationTimeType, localComputationTimeUnit,
				localComputationTimeDescription);
	}

	/*
	 * ==================== Private Helper Class ====================
	 */

	class Property {
		private String name;
		private String value;
		private String type;
		private String unit;
		private String description;

		public Property(String name, String value, String type, String unit,
				String description) {
			super();
			this.name = name;
			if (this.name.equals("local-computation-time")) {
				long c = Long.parseLong(value);
				c = (long) (c * SCALE);
				this.value = new String("" + c);
			} else {
				this.value = value;
			}
			this.type = type;
			this.unit = unit;
			this.description = description;
		}

		public String getName() {
			return name;
		}

		public String getValue() {
			return value;
		}

		public String getType() {
			return type;
		}

		public String getUnit() {
			return unit;
		}

		public String getDescription() {
			return description;
		}

	}

	class MethodPerformance {
		private long minTime;
		private long maxTime;
		private long totalTime;
		private long invocationCount;

		public MethodPerformance(long minTime, long maxTime, long totalTime,
				long invocationCount) {
			super();
			this.minTime = (long) (minTime * SCALE);
			this.maxTime = (long) (maxTime * SCALE);
			this.totalTime = (long) (totalTime * SCALE);
			this.invocationCount = invocationCount;
		}

		public long getMinTime() {
			return minTime;
		}

		public long getMaxTime() {
			return maxTime;
		}

		public long getTotalTime() {
			return totalTime;
		}

		public long getInvocationCount() {
			return invocationCount;
		}

	}

	/**
	 * Representation of a method of a business interface.
	 * 
	 * @author thomas vogel
	 * @since 0.1
	 * @version 0.1
	 * 
	 */
	class Method {

		private String fullyQualifiedInterface;
		private String signature;

		/**
		 * Constructor.
		 * 
		 * @param fullyQualifiedInterface
		 *            the fully qualified name of the interface to which the
		 *            method belongs
		 * @param signature
		 *            the method signature
		 */
		public Method(String fullyQualifiedInterface, String signature) {
			super();
			this.fullyQualifiedInterface = fullyQualifiedInterface;
			this.signature = signature;
		}

		public String getSignature() {
			return signature;
		}

		public String getFullyQualifiedInterface() {
			return fullyQualifiedInterface;
		}

	}

}
