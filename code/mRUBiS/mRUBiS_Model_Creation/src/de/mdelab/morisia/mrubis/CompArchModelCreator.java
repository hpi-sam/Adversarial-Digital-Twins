package de.mdelab.morisia.mrubis;

import java.io.IOException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;

import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.ResourceSet;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;

import de.hpi.sam.rubis.authservice.AuthenticationService;
import de.hpi.sam.rubis.bidandbuy.BidService;
import de.hpi.sam.rubis.bidandbuy.BuyNowService;
import de.hpi.sam.rubis.filter.ItemFilter;
import de.hpi.sam.rubis.inventorymgmt.InventoryService;
import de.hpi.sam.rubis.itemmgmt.BrowseCategoriesService;
import de.hpi.sam.rubis.itemmgmt.ItemRegistrationService;
import de.hpi.sam.rubis.persistenceservice.BusinessObjectsPersistenceService;
import de.hpi.sam.rubis.queryservice.BasicQueryService;
import de.hpi.sam.rubis.queryservice.QueryService;
import de.hpi.sam.rubis.reputationservice.ReputationService;
import de.hpi.sam.rubis.usermgmt.AboutMeService;
import de.hpi.sam.rubis.usermgmt.BrowseRegionsService;
import de.hpi.sam.rubis.usermgmt.UserRegistrationService;
import de.hpi.sam.rubis.usermgmt.ViewUserInfoService;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.Cloud;
import de.mdelab.morisia.comparch.ComparchFactory;
import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.Connector;
import de.mdelab.morisia.comparch.Interface;
import de.mdelab.morisia.comparch.InterfaceType;
import de.mdelab.morisia.comparch.MethodSpecification;
import de.mdelab.morisia.comparch.MonitoredProperty;
import de.mdelab.morisia.comparch.ParameterType;
import de.mdelab.morisia.comparch.PerformanceStats;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Replica;
import de.mdelab.morisia.comparch.RequiredInterface;
import de.mdelab.morisia.comparch.Server;
import de.mdelab.morisia.comparch.Cloud;
import de.mdelab.morisia.comparch.Tenant;
import de.mdelab.morisia.mrubis.CompArchPerformanceDataGenerator.MethodPerformance;
import de.mdelab.morisia.mrubis.CompArchPerformanceDataGenerator.Property;

public class CompArchModelCreator {

	private static String modelFileURI = "./model/mRUBiS-1Shop.comparch";
	// whether the reliability of a component type should be set or not to the
	// default.
	private static boolean SET_DEFAULT_RELIABILITY = false;
	// the default reliability
	private static double DEFAULT_RELIABILTY = 1.00;

	// whether performance data should be added to the model
	private static boolean ADD_PERFORMANCE_DATA = false;

	// implemented in mRUBiS
	private static boolean ADD_FACEBOOK_AHTENTICATION = true;
	// not implemented in mRUBiS
	private static boolean ADD_GOOGLE_AUTHENTICATION = true;
	// not implemented in mRUBiS
	private static boolean ADD_TWITTER_AHTENTICATION = true;

	public static void main(String[] args) {
		configure();

		int numberOfTenants = 1;
		int numberOfServers = 0;
	
		
		Architecture architecture = createModel(numberOfTenants , numberOfServers);
		saveModelResource(architecture, modelFileURI);
		

		if (ADD_PERFORMANCE_DATA) {
			// tests and outputs the added performance data
			testArchitecture(modelFileURI);
		}

	}

	public static Architecture createModel(int numberOfTenants, int numberOfServers) {
		Architecture architecture = ComparchFactory.eINSTANCE
				.createArchitecture();
		//architecture.setUid(EcoreUtil.generateUUID());
		architecture.setName("mRUBiS");
		
		
		
		createInterfaceTypes(architecture);
		createComponentTypes(architecture);

		if (ADD_FACEBOOK_AHTENTICATION) {
			createExternalAuthenticationServiceComponentType(
					architecture,
					externalFacebookAuthenticationSerivce_ComponentType,
					"Facebook Authentication Service", // originally named
														// "External Authentication Service".
					"http://localhost:8080/FacebookAuthenticationService/FbAuthenticationServiceImpl?wsdl",
					"http://authentication.facebook.com/",
					"FacebookAuthenticationService");
		}
		if (ADD_GOOGLE_AUTHENTICATION) {
			createExternalAuthenticationServiceComponentType(
					architecture,
					externalGoogleAuthenticationSerivce_ComponentType,
					"Google Authentication Service",
					"http://localhost:8080/GoogleAuthenticationService/GoogleAuthenticationServiceImpl?wsdl",
					"http://authentication.google.com/",
					"GoogleAuthenticationService");
		}

		if (ADD_TWITTER_AHTENTICATION) {
			createExternalAuthenticationServiceComponentType(
					architecture,
					externalTwitterAuthenticationSerivce_ComponentType,
					"Twitter Authentication Service",
					"http://localhost:8080/TwitterAuthenticationService/TwitterAuthenticationServiceImpl?wsdl",
					"http://authentication.twitter.com/",
					"TwitterAuthenticationService");
		}

		boolean flag = true;
		Cloud cloud = instantiateCloud(architecture);
		for (int i = 0; i < numberOfTenants; i++) {
			Tenant tenant = instantiateTenant(architecture, "mRUBiS #"
					+ (i + 1));

			if (ADD_PERFORMANCE_DATA) {
				attachStatistics(tenant);
				if (flag) {
					// next run uses statistics scaled up by 5 %.
					CompArchPerformanceDataGenerator.SCALE = CompArchPerformanceDataGenerator.SCALE + 0.05;
					flag = false;
				} else {
					CompArchPerformanceDataGenerator.SCALE = CompArchPerformanceDataGenerator.SCALE - 0.05;
					flag = true;
				}
			}
		}
		
		for (int i = 0; i < numberOfServers; i++) {
			Server server = instantiateServers(architecture,cloud, "server #"
					+ (i + 1));
		}
		
	
		

		return architecture;
	}

	private static Cloud instantiateCloud(Architecture architecture) {
		Cloud cloud = ComparchFactory.eINSTANCE.createCloud();
		cloud.setName("AmazonCloud");
		//shop.setUid(shopName + "_" + EcoreUtil.generateUUID());
		cloud.setArchitecture(architecture);
		return cloud;
	}

	private static Server instantiateServers(Architecture architecture, Cloud cloud,
			String serverName) {
		
		//Tenant shop = ComparchFactory.eINSTANCE.createTenant();
		Server server = ComparchFactory.eINSTANCE.createServer();
		
		//shop.setName(shopName);
		server.setName(serverName);
		
		
		//shop.setArchitecture(architecture);
		 server.setCloud(cloud);
		  
		 // System.out.println(server.getCloud());
		return server;}
	

	public static void testArchitecture(String modelFileURI) {
		EObject elem = loadModelFromXmi(modelFileURI);
		if (elem instanceof Architecture) {
			Architecture architecture = (Architecture) elem;
			for (Tenant shop : architecture.getTenants()) {
				System.out.println("Shop: " + shop.getName());

				// check filter order in the pipe based on the slopes
				// -- find the Query Service
				Component queryService = null;
				OUTERLOOP: for (ComponentType ct : architecture
						.getComponentTypes()) {
					if (ct.getName().equals("Query Service")) {
						List<Component> ctInstances = ct.getInstances();
						for (Component ctInstance : ctInstances) {
							if (ctInstance.getTenant().equals(shop)) {
								queryService = ctInstance;
								break OUTERLOOP;
							}
						}
					}
				}
				// -- get first filter component
				Component filter = null;
				for (RequiredInterface reqInterface : queryService
						.getRequiredInterfaces()) {
					if (reqInterface
							.getType()
							.getFullyQualifiedName()
							.equals(itemFilter_InterfaceType
									.getFullyQualifiedName())) {
						filter = reqInterface.getConnector().getTarget()
								.getComponent();
						break;
					}
				}

				Double[] slopes = new Double[10];

				int i = 1;
				while (filter != null) {
					MonitoredProperty s = null;
					MonitoredProperty c = null;
					for (MonitoredProperty prop : filter
							.getMonitoredProperties()) {
						if (prop.getName().equals("selection-rate")) {
							s = prop;
						} else if (prop.getName().equals(
								"local-computation-time")) {
							c = prop;
						}
					}

					if (i < 10) {
						System.out.print(" ");
					}
					double slope = Double.parseDouble(s.getValue())
							/ Long.parseLong(c.getValue());

					slopes[i - 1] = slope;
					System.out.println(i + ". filter has the slope = " + slope
							+ "\t\t\t Filter uid = " + filter.getUid());

					if (filter.getRequiredInterfaces().get(0).getConnector() != null) {
						filter = filter.getRequiredInterfaces().get(0)
								.getConnector().getTarget().getComponent();
					} else {
						filter = null;
					}
					i++;
				}

				for (int j = 0; j < slopes.length - 1; j++) {
					// System.out.println(slopes[j] + " >= " + slopes[j + 1]
					// + " is " + ((slopes[j] - slopes[j + 1]) >= 0));
					if ((slopes[j] - slopes[j + 1]) < 0) {
						System.err.println("Filters " + j + " and " + (j + 1)
								+ " are not well-ordered!");
					}
				}

			}
			for (Server server : architecture.getCloud().getServers()) {
				System.out.println("\n\n  Server"			+ ": " + server.getName());

		}
		}
	}

	/*
	 * ========================================================
	 */
	/*
	 * InterfaceTypes
	 */

	private static InterfaceType authenticationService_InterfaceType,
			bidService_InterfaceType, buyNowService_InterfaceType,
			itemFilter_InterfaceType, inventoryService_InterfaceType,
			browseCategoriesService_InterfaceType,
			itemRegistrationService_InterfaceType,
			businessObjectsPersistenceService_InterfaceType,
			basicQueryService_InterfaceType, queryService_InterfaceType,
			reputationService_InterfaceType, aboutMeService_InterfaceType,
			browseRegionService_InterfaceType,
			userRegistrationService_InterfaceType,
			viewUserInfoService_InterfaceType;

	private static void createInterfaceTypes(Architecture architecture) {
		authenticationService_InterfaceType = createAndAddInterfaceType(
				AuthenticationService.class, architecture);
		bidService_InterfaceType = createAndAddInterfaceType(BidService.class,
				architecture);
		buyNowService_InterfaceType = createAndAddInterfaceType(
				BuyNowService.class, architecture);
		itemFilter_InterfaceType = createAndAddInterfaceType(ItemFilter.class,
				architecture);
		inventoryService_InterfaceType = createAndAddInterfaceType(
				InventoryService.class, architecture);
		browseCategoriesService_InterfaceType = createAndAddInterfaceType(
				BrowseCategoriesService.class, architecture);
		itemRegistrationService_InterfaceType = createAndAddInterfaceType(
				ItemRegistrationService.class, architecture);
		businessObjectsPersistenceService_InterfaceType = createAndAddInterfaceType(
				BusinessObjectsPersistenceService.class, architecture);
		queryService_InterfaceType = createAndAddInterfaceType(
				QueryService.class, architecture);
		basicQueryService_InterfaceType = createAndAddInterfaceType(
				BasicQueryService.class, architecture);
		reputationService_InterfaceType = createAndAddInterfaceType(
				ReputationService.class, architecture);
		aboutMeService_InterfaceType = createAndAddInterfaceType(
				AboutMeService.class, architecture);
		browseRegionService_InterfaceType = createAndAddInterfaceType(
				BrowseRegionsService.class, architecture);
		userRegistrationService_InterfaceType = createAndAddInterfaceType(
				UserRegistrationService.class, architecture);
		viewUserInfoService_InterfaceType = createAndAddInterfaceType(
				ViewUserInfoService.class, architecture);
	}

	private static InterfaceType createAndAddInterfaceType(
			Class<?> interfaceClass, Architecture architecture) {
		InterfaceType interfaceType = ComparchFactory.eINSTANCE
				.createInterfaceType();
		interfaceType.setFullyQualifiedName(interfaceClass.getCanonicalName());
		// interfaceType.setUid("InterfaceType_"
		// + interfaceType.getFullyQualifiedName()
		// + EcoreUtil.generateUUID());
		for (Method method : interfaceClass.getMethods()) {
			MethodSpecification methodSpecification = ComparchFactory.eINSTANCE
					.createMethodSpecification();
			//methodSpecification.setUid(EcoreUtil.generateUUID());
			String signature = method.getName() + "(";
			boolean hasParams = method.getParameterTypes().length > 0;
			for (Class<?> paramType : method.getParameterTypes()) {
				signature = signature + paramType.getCanonicalName() + ",";
			}
			if (hasParams) {
				signature = signature.substring(0, signature.length() - 1);
			}
			signature = signature + ")";
			methodSpecification.setSignature(signature);
			interfaceType.getMethodSpecification().add(methodSpecification);
		}
		architecture.getInterfaceTypes().add(interfaceType);
		return interfaceType;
	}

	/*
	 * ========================================================
	 */
	/*
	 * ComponentTypes
	 */

	private static ComponentType userManagementService_ComponentType,
			itemManagementService_ComponentType,
			authenticationService_ComponentType,
			bidAndBuyService_ComponentType, inventoryService_ComponentType,
			reputationService_ComponentType, persistenceService_ComponentType,
			queryService_ComponentType,
			lastSecondSalesItemFilter_ComponentType,
			pastSalesItemFilter_ComponentType, buyNowItemFilter_ComponentType,
			availabilityItemFilter_ComponentType,
			regionItemFilter_ComponentType,
			sellerReputationItemFilter_ComponentType,
			commentItemFilter_ComponentType, categoryItemFilter_ComponentType,
			recommendationItemFilter_ComponentType,
			futureSalesItemFilter_ComponentType;
	private static ComponentType externalFacebookAuthenticationSerivce_ComponentType;
	private static ComponentType externalGoogleAuthenticationSerivce_ComponentType;
	private static ComponentType externalTwitterAuthenticationSerivce_ComponentType;

	private static void createComponentTypes(Architecture architecture) {
		userManagementService_ComponentType = createComponentTypeWithName(
				MRubisNames.userManagementService_ComponentType_Name,
				architecture);
		userManagementService_ComponentType.getProvidedInterfaceTypes().add(
				aboutMeService_InterfaceType);
		userManagementService_ComponentType.getProvidedInterfaceTypes().add(
				browseRegionService_InterfaceType);
		userManagementService_ComponentType.getProvidedInterfaceTypes().add(
				userRegistrationService_InterfaceType);
		userManagementService_ComponentType.getProvidedInterfaceTypes().add(
				viewUserInfoService_InterfaceType);
		userManagementService_ComponentType.getRequiredInterfaceTypes().add(
				authenticationService_InterfaceType);
		userManagementService_ComponentType.getRequiredInterfaceTypes().add(
				queryService_InterfaceType);
		userManagementService_ComponentType.getRequiredInterfaceTypes().add(
				basicQueryService_InterfaceType);
		userManagementService_ComponentType.getRequiredInterfaceTypes().add(
				businessObjectsPersistenceService_InterfaceType);
		ParameterType privacyLevel_parameterType = createParameterType(
				"privacy-level", String.class.getCanonicalName(), "LOW");
		userManagementService_ComponentType.getParameterTypes().add(
				privacyLevel_parameterType);

		itemManagementService_ComponentType = createComponentTypeWithName(
				MRubisNames.itemManagementService_ComponentType_Name,
				architecture);
		itemManagementService_ComponentType.getProvidedInterfaceTypes().add(
				browseCategoriesService_InterfaceType);
		itemManagementService_ComponentType.getProvidedInterfaceTypes().add(
				itemRegistrationService_InterfaceType);
		itemManagementService_ComponentType.getRequiredInterfaceTypes().add(
				basicQueryService_InterfaceType);
		itemManagementService_ComponentType.getRequiredInterfaceTypes().add(
				queryService_InterfaceType);
		itemManagementService_ComponentType.getRequiredInterfaceTypes().add(
				authenticationService_InterfaceType);
		itemManagementService_ComponentType.getRequiredInterfaceTypes().add(
				businessObjectsPersistenceService_InterfaceType);

		authenticationService_ComponentType = createComponentTypeWithName(
				MRubisNames.authenticationService_ComponentType_Name,
				architecture);
		authenticationService_ComponentType.getProvidedInterfaceTypes().add(
				authenticationService_InterfaceType);
		authenticationService_ComponentType.getRequiredInterfaceTypes().add(
				basicQueryService_InterfaceType);

		bidAndBuyService_ComponentType = createComponentTypeWithName(
				MRubisNames.bidAndBuyService_ComponentType_Name, architecture);
		bidAndBuyService_ComponentType.getProvidedInterfaceTypes().add(
				bidService_InterfaceType);
		bidAndBuyService_ComponentType.getProvidedInterfaceTypes().add(
				buyNowService_InterfaceType);
		bidAndBuyService_ComponentType.getRequiredInterfaceTypes().add(
				basicQueryService_InterfaceType);
		bidAndBuyService_ComponentType.getRequiredInterfaceTypes().add(
				queryService_InterfaceType);
		bidAndBuyService_ComponentType.getRequiredInterfaceTypes().add(
				authenticationService_InterfaceType);
		bidAndBuyService_ComponentType.getRequiredInterfaceTypes().add(
				inventoryService_InterfaceType);
		bidAndBuyService_ComponentType.getRequiredInterfaceTypes().add(
				businessObjectsPersistenceService_InterfaceType);

		inventoryService_ComponentType = createComponentTypeWithName(
				MRubisNames.inventoryService_ComponentType_Name, architecture);
		inventoryService_ComponentType.getProvidedInterfaceTypes().add(
				inventoryService_InterfaceType);
		inventoryService_ComponentType.getRequiredInterfaceTypes().add(
				queryService_InterfaceType);
		inventoryService_ComponentType.getRequiredInterfaceTypes().add(
				businessObjectsPersistenceService_InterfaceType);

		reputationService_ComponentType = createComponentTypeWithName(
				MRubisNames.reputationService_ComponentType_Name, architecture);
		reputationService_ComponentType.getProvidedInterfaceTypes().add(
				reputationService_InterfaceType);
		reputationService_ComponentType.getRequiredInterfaceTypes().add(
				basicQueryService_InterfaceType);
		reputationService_ComponentType.getRequiredInterfaceTypes().add(
				queryService_InterfaceType);
		reputationService_ComponentType.getRequiredInterfaceTypes().add(
				authenticationService_InterfaceType);
		reputationService_ComponentType.getRequiredInterfaceTypes().add(
				businessObjectsPersistenceService_InterfaceType);

		persistenceService_ComponentType = createComponentTypeWithName(
				MRubisNames.persistenceService_ComponentType_Name, architecture);
		persistenceService_ComponentType.getProvidedInterfaceTypes().add(
				businessObjectsPersistenceService_InterfaceType);

		queryService_ComponentType = createComponentTypeWithName(
				MRubisNames.queryService_ComponentType_Name, architecture);
		queryService_ComponentType.getProvidedInterfaceTypes().add(
				basicQueryService_InterfaceType);
		queryService_ComponentType.getProvidedInterfaceTypes().add(
				queryService_InterfaceType);
		queryService_ComponentType.getRequiredInterfaceTypes().add(
				itemFilter_InterfaceType);

		lastSecondSalesItemFilter_ComponentType = createItemFilterComponentTypeWithName(
				MRubisNames.lastSecondSalesItemFilter_ComponentType_Name,
				architecture);
		ParameterType lastSeconds_ParameterType = createParameterType(
				"last-seconds", Integer.class.getCanonicalName(), "300");
		lastSecondSalesItemFilter_ComponentType.getParameterTypes().add(
				lastSeconds_ParameterType);

		pastSalesItemFilter_ComponentType = createItemFilterComponentTypeWithName(
				MRubisNames.pastSalesItemFilter_ComponentType_Name,
				architecture);

		buyNowItemFilter_ComponentType = createItemFilterComponentTypeWithName(
				MRubisNames.buyNowItemFilter_ComponentType_Name, architecture);

		availabilityItemFilter_ComponentType = createItemFilterComponentTypeWithName(
				MRubisNames.availabilityItemFilter_ComponentType_Name,
				architecture);

		regionItemFilter_ComponentType = createItemFilterComponentTypeWithName(
				MRubisNames.regionItemFilter_ComponentType_Name, architecture);

		sellerReputationItemFilter_ComponentType = createItemFilterComponentTypeWithName(
				MRubisNames.sellerReputationItemFilter_ComponentType_Name,
				architecture);
		ParameterType reputationThreshold_ParameterType = createParameterType(
				"reputation-threshold", Integer.class.getCanonicalName(), "5");
		sellerReputationItemFilter_ComponentType.getParameterTypes().add(
				reputationThreshold_ParameterType);

		commentItemFilter_ComponentType = createItemFilterComponentTypeWithName(
				MRubisNames.commentItemFilter_ComponentType_Name, architecture);
		ParameterType ratingThreshold_ParameterType = createParameterType(
				"rating-threshold", Integer.class.getCanonicalName(), "5");
		commentItemFilter_ComponentType.getParameterTypes().add(
				ratingThreshold_ParameterType);

		categoryItemFilter_ComponentType = createItemFilterComponentTypeWithName(
				MRubisNames.categoryItemFilter_ComponentType_Name, architecture);
		ParameterType filteredCategories_ParameterType = createParameterType(
				"filtered-categories", String.class.getCanonicalName(),
				"Coins,Jewelry");
		categoryItemFilter_ComponentType.getParameterTypes().add(
				filteredCategories_ParameterType);

		recommendationItemFilter_ComponentType = createItemFilterComponentTypeWithName(
				MRubisNames.recommendationItemFilter_ComponentType_Name,
				architecture);

		futureSalesItemFilter_ComponentType = createItemFilterComponentTypeWithName(
				MRubisNames.futureSalesItemFilter_ComponentType_Name,
				architecture);
		ParameterType daysToRun_ParameterType = createParameterType(
				"days-to-run", Integer.class.getCanonicalName(), "14");
		futureSalesItemFilter_ComponentType.getParameterTypes().add(
				daysToRun_ParameterType);
	}

	private static void createExternalAuthenticationServiceComponentType(
			Architecture architecture, ComponentType componentType,
			String componentTypeName, String serviceURI_Value,
			String serviceNS_Value, String serviceNsLocalPartName_Value) {
		componentType = createComponentTypeWithName(componentTypeName,
				architecture);
		componentType.getProvidedInterfaceTypes().add(
				authenticationService_InterfaceType);
		componentType.getRequiredInterfaceTypes().add(
				basicQueryService_InterfaceType);
		ParameterType serviceURI = createParameterType("ServiceURI",
				String.class.getCanonicalName(), serviceURI_Value);
		componentType.getParameterTypes().add(serviceURI);
		ParameterType serviceNS = createParameterType("ServiceNS",
				String.class.getCanonicalName(), serviceNS_Value);
		componentType.getParameterTypes().add(serviceNS);
		ParameterType serviceNsLocalPartName = createParameterType(
				"ServiceNsLocalPartName", String.class.getCanonicalName(),
				serviceNsLocalPartName_Value);
		componentType.getParameterTypes().add(serviceNsLocalPartName);
	}

	private static ComponentType createItemFilterComponentTypeWithName(
			String name, Architecture architecture) {
		ComponentType itemFilter_ComponentType = createComponentTypeWithName(
				name, architecture);
		itemFilter_ComponentType.getProvidedInterfaceTypes().add(
				itemFilter_InterfaceType);
		itemFilter_ComponentType.getRequiredInterfaceTypes().add(
				itemFilter_InterfaceType);
		itemFilter_ComponentType.getParameterTypes().add(
				createSelectionRateThresholdParameterType());
		itemFilter_ComponentType.getParameterTypes().add(
				createComputationTimeThresholdParameterType());
		return itemFilter_ComponentType;
	}

	private static ComponentType createComponentTypeWithName(String name,
			Architecture architecture) {
		ComponentType componentType = ComparchFactory.eINSTANCE
				.createComponentType();
		componentType.setName(name);
		// componentType
		// .setUid("ComponentType_" + name + EcoreUtil.generateUUID());
		architecture.getComponentTypes().add(componentType);
		if (SET_DEFAULT_RELIABILITY) {
			componentType.setReliability(DEFAULT_RELIABILTY);
		}
		return componentType;
	}

	private static ParameterType createSelectionRateThresholdParameterType() {
		return createParameterType("selection-rate-threshold",
				Double.class.getCanonicalName(), "0.05");
	}

	private static ParameterType createComputationTimeThresholdParameterType() {
		return createParameterType("computation-time-threshold",
				Double.class.getCanonicalName(), "0.05");
	}

	private static ParameterType createParameterType(String name, String type,
			String defaultValue) {
		ParameterType parameterType = ComparchFactory.eINSTANCE
				.createParameterType();
		// parameterType
		// .setUid("ParameterType_" + name + EcoreUtil.generateUUID());
		parameterType.setName(name);
		parameterType.setType(type);
		parameterType.setDefaultValue(defaultValue);
		return parameterType;
	}

	/*
	 * ========================================================
	 */
	/*
	 * Components
	 */
	private static Tenant instantiateTenant(Architecture architecture,
			String shopName) {
		Tenant shop = ComparchFactory.eINSTANCE.createTenant();
		shop.setName(shopName);
		//shop.setUid(shopName + "_" + EcoreUtil.generateUUID());
		shop.setArchitecture(architecture);

		List<Component> components = new ArrayList<Component>(18);
		Component userManagementService_Component = userManagementService_ComponentType
				.instantiate();
		components.add(userManagementService_Component);
		ProvidedInterface userManagementService__aboutMeService_ProvInt = getInterfaceOfType(
				userManagementService_Component.getProvidedInterfaces(),
				aboutMeService_InterfaceType);
		ProvidedInterface userManagementService__viewUserInfoService_ProvInt = getInterfaceOfType(
				userManagementService_Component.getProvidedInterfaces(),
				viewUserInfoService_InterfaceType);
		ProvidedInterface userManagementService__browseRegionsService_ProvInt = getInterfaceOfType(
				userManagementService_Component.getProvidedInterfaces(),
				browseRegionService_InterfaceType);
		ProvidedInterface userManagementService__userRegistrationService_ProvInt = getInterfaceOfType(
				userManagementService_Component.getProvidedInterfaces(),
				userRegistrationService_InterfaceType);
		RequiredInterface userManagementService__basicQueryService_ReqInt = getInterfaceOfType(
				userManagementService_Component.getRequiredInterfaces(),
				basicQueryService_InterfaceType);
		RequiredInterface userManagementService__queryService_ReqInt = getInterfaceOfType(
				userManagementService_Component.getRequiredInterfaces(),
				queryService_InterfaceType);
		RequiredInterface userManagementService__authenticationService_ReqInt = getInterfaceOfType(
				userManagementService_Component.getRequiredInterfaces(),
				authenticationService_InterfaceType);
		RequiredInterface userManagementService__businessObjectPersistenceService_ReqInt = getInterfaceOfType(
				userManagementService_Component.getRequiredInterfaces(),
				businessObjectsPersistenceService_InterfaceType);

		Component itemManagementService_Component = itemManagementService_ComponentType
				.instantiate();
		components.add(itemManagementService_Component);
		ProvidedInterface itemManagementService__browseCategoriesService_ProvInt = getInterfaceOfType(
				itemManagementService_Component.getProvidedInterfaces(),
				browseCategoriesService_InterfaceType);
		ProvidedInterface itemManagementService__itemRegistrationService_ProvInt = getInterfaceOfType(
				itemManagementService_Component.getProvidedInterfaces(),
				itemRegistrationService_InterfaceType);
		RequiredInterface itemManagementService__basicQueryService_ReqInt = getInterfaceOfType(
				itemManagementService_Component.getRequiredInterfaces(),
				basicQueryService_InterfaceType);
		RequiredInterface itemManagementService__queryService_ReqInt = getInterfaceOfType(
				itemManagementService_Component.getRequiredInterfaces(),
				queryService_InterfaceType);
		RequiredInterface itemManagementService__authenticationService_ReqInt = getInterfaceOfType(
				itemManagementService_Component.getRequiredInterfaces(),
				authenticationService_InterfaceType);
		RequiredInterface itemManagementService__businessObjectPersistenceService_ReqInt = getInterfaceOfType(
				itemManagementService_Component.getRequiredInterfaces(),
				businessObjectsPersistenceService_InterfaceType);

		Component authenticationService_Component = authenticationService_ComponentType
				.instantiate();
		components.add(authenticationService_Component);
		ProvidedInterface authenticationService__authenticationService_ProvInt = getInterfaceOfType(
				authenticationService_Component.getProvidedInterfaces(),
				authenticationService_InterfaceType);
		RequiredInterface authenticationService__basicQueryService_ReqInt = getInterfaceOfType(
				authenticationService_Component.getRequiredInterfaces(),
				basicQueryService_InterfaceType);

		Component bidAndBuyService_Component = bidAndBuyService_ComponentType
				.instantiate();
		components.add(bidAndBuyService_Component);
		ProvidedInterface bidAndBuyService__bidService_ProvInt = getInterfaceOfType(
				bidAndBuyService_Component.getProvidedInterfaces(),
				bidService_InterfaceType);
		ProvidedInterface bidAndBuyService__buyNowService_ProvInt = getInterfaceOfType(
				bidAndBuyService_Component.getProvidedInterfaces(),
				buyNowService_InterfaceType);
		RequiredInterface bidAndBuyService__basicQueryService_ReqInt = getInterfaceOfType(
				bidAndBuyService_Component.getRequiredInterfaces(),
				basicQueryService_InterfaceType);
		RequiredInterface bidAndBuyService__queryService_ReqInt = getInterfaceOfType(
				bidAndBuyService_Component.getRequiredInterfaces(),
				queryService_InterfaceType);
		RequiredInterface bidAndBuyService__authenticationService_ReqInt = getInterfaceOfType(
				bidAndBuyService_Component.getRequiredInterfaces(),
				authenticationService_InterfaceType);
		RequiredInterface bidAndBuyService__businessObjectPersistenceService_ReqInt = getInterfaceOfType(
				bidAndBuyService_Component.getRequiredInterfaces(),
				businessObjectsPersistenceService_InterfaceType);
		RequiredInterface bidAndBuyService__inventoryService_ReqInt = getInterfaceOfType(
				bidAndBuyService_Component.getRequiredInterfaces(),
				inventoryService_InterfaceType);

		Component inventoryService_Component = inventoryService_ComponentType
				.instantiate();
		components.add(inventoryService_Component);
		ProvidedInterface inventoryService__inventoryService_ProvInt = getInterfaceOfType(
				inventoryService_Component.getProvidedInterfaces(),
				inventoryService_InterfaceType);
		RequiredInterface inventoryService__businessObjectPersistenceService_ReqInt = getInterfaceOfType(
				inventoryService_Component.getRequiredInterfaces(),
				businessObjectsPersistenceService_InterfaceType);
		RequiredInterface inventoryService__queryService_ReqInt = getInterfaceOfType(
				inventoryService_Component.getRequiredInterfaces(),
				queryService_InterfaceType);

		Component reputationService_Component = reputationService_ComponentType
				.instantiate();
		components.add(reputationService_Component);
		ProvidedInterface reputationService__reputationService_ProvInt = getInterfaceOfType(
				reputationService_Component.getProvidedInterfaces(),
				reputationService_InterfaceType);
		RequiredInterface reputationService__queryService_ReqInt = getInterfaceOfType(
				reputationService_Component.getRequiredInterfaces(),
				queryService_InterfaceType);
		RequiredInterface reputationService__basicQueryService_ReqInt = getInterfaceOfType(
				reputationService_Component.getRequiredInterfaces(),
				basicQueryService_InterfaceType);
		RequiredInterface reputationService__authenticationService_ReqInt = getInterfaceOfType(
				reputationService_Component.getRequiredInterfaces(),
				authenticationService_InterfaceType);
		RequiredInterface reputationService__businessObjectPersistenceService_ReqInt = getInterfaceOfType(
				reputationService_Component.getRequiredInterfaces(),
				businessObjectsPersistenceService_InterfaceType);

		Component persistenceService_Component = persistenceService_ComponentType
				.instantiate();
		components.add(persistenceService_Component);
		ProvidedInterface persistenceService__businessObjectPersistenceService_ProvInt = getInterfaceOfType(
				persistenceService_Component.getProvidedInterfaces(),
				businessObjectsPersistenceService_InterfaceType);

		Component queryService_Component = queryService_ComponentType
				.instantiate();
		components.add(queryService_Component);
		ProvidedInterface queryService__basicQueryService_ProvInt = getInterfaceOfType(
				queryService_Component.getProvidedInterfaces(),
				basicQueryService_InterfaceType);
		ProvidedInterface queryService__queryService_ProvInt = getInterfaceOfType(
				queryService_Component.getProvidedInterfaces(),
				queryService_InterfaceType);
		RequiredInterface queryService__itemFilter_ReqInt = getInterfaceOfType(
				queryService_Component.getRequiredInterfaces(),
				itemFilter_InterfaceType);

		Component lastSecondSalesItemFilter_Component = lastSecondSalesItemFilter_ComponentType
				.instantiate();
		components.add(lastSecondSalesItemFilter_Component);
		ProvidedInterface lastSecondSalesItemFilter__itemFilter_ProvInt = getInterfaceOfType(
				lastSecondSalesItemFilter_Component.getProvidedInterfaces(),
				itemFilter_InterfaceType);
		RequiredInterface lastSecondSalesItemFilter__itemFilter_ReqInt = getInterfaceOfType(
				lastSecondSalesItemFilter_Component.getRequiredInterfaces(),
				itemFilter_InterfaceType);

		Component pastSalesItemFilter_Component = pastSalesItemFilter_ComponentType
				.instantiate();
		components.add(pastSalesItemFilter_Component);
		ProvidedInterface pastSalesItemFilter__itemFilter_ProvInt = getInterfaceOfType(
				pastSalesItemFilter_Component.getProvidedInterfaces(),
				itemFilter_InterfaceType);
		RequiredInterface pastSalesItemFilter__itemFilter_ReqInt = getInterfaceOfType(
				pastSalesItemFilter_Component.getRequiredInterfaces(),
				itemFilter_InterfaceType);

		Component buyNowItemFilter_Component = buyNowItemFilter_ComponentType
				.instantiate();
		components.add(buyNowItemFilter_Component);
		ProvidedInterface buyNowItemFilter__itemFilter_ProvInt = getInterfaceOfType(
				buyNowItemFilter_Component.getProvidedInterfaces(),
				itemFilter_InterfaceType);
		RequiredInterface buyNowItemFilter__itemFilter_ReqInt = getInterfaceOfType(
				buyNowItemFilter_Component.getRequiredInterfaces(),
				itemFilter_InterfaceType);

		Component availabilityItemFilter_Component = availabilityItemFilter_ComponentType
				.instantiate();
		components.add(availabilityItemFilter_Component);
		ProvidedInterface availabilityItemFilter__itemFilter_ProvInt = getInterfaceOfType(
				availabilityItemFilter_Component.getProvidedInterfaces(),
				itemFilter_InterfaceType);
		RequiredInterface availabilityItemFilter__itemFilter_ReqInt = getInterfaceOfType(
				availabilityItemFilter_Component.getRequiredInterfaces(),
				itemFilter_InterfaceType);

		Component regionItemFilter_Component = regionItemFilter_ComponentType
				.instantiate();
		components.add(regionItemFilter_Component);
		ProvidedInterface regionItemFilter__itemFilter_ProvInt = getInterfaceOfType(
				regionItemFilter_Component.getProvidedInterfaces(),
				itemFilter_InterfaceType);
		RequiredInterface regionItemFilter__itemFilter_ReqInt = getInterfaceOfType(
				regionItemFilter_Component.getRequiredInterfaces(),
				itemFilter_InterfaceType);

		Component sellerReputationItemFilter_Component = sellerReputationItemFilter_ComponentType
				.instantiate();
		components.add(sellerReputationItemFilter_Component);
		ProvidedInterface sellerReputationItemFilter__itemFilter_ProvInt = getInterfaceOfType(
				sellerReputationItemFilter_Component.getProvidedInterfaces(),
				itemFilter_InterfaceType);
		RequiredInterface sellerReputationItemFilter__itemFilter_ReqInt = getInterfaceOfType(
				sellerReputationItemFilter_Component.getRequiredInterfaces(),
				itemFilter_InterfaceType);

		Component commentItemFilter_Component = commentItemFilter_ComponentType
				.instantiate();
		components.add(commentItemFilter_Component);
		ProvidedInterface commentItemFilter__itemFilter_ProvInt = getInterfaceOfType(
				commentItemFilter_Component.getProvidedInterfaces(),
				itemFilter_InterfaceType);
		RequiredInterface commentItemFilter__itemFilter_ReqInt = getInterfaceOfType(
				commentItemFilter_Component.getRequiredInterfaces(),
				itemFilter_InterfaceType);

		Component categoryItemFilter_Component = categoryItemFilter_ComponentType
				.instantiate();
		components.add(categoryItemFilter_Component);
		ProvidedInterface categoryItemFilter__itemFilter_ProvInt = getInterfaceOfType(
				categoryItemFilter_Component.getProvidedInterfaces(),
				itemFilter_InterfaceType);
		RequiredInterface categoryItemFilter__itemFilter_ReqInt = getInterfaceOfType(
				categoryItemFilter_Component.getRequiredInterfaces(),
				itemFilter_InterfaceType);

		Component recommendationItemFilter_Component = recommendationItemFilter_ComponentType
				.instantiate();
		components.add(recommendationItemFilter_Component);
		ProvidedInterface recommendationItemFilter__itemFilter_ProvInt = getInterfaceOfType(
				recommendationItemFilter_Component.getProvidedInterfaces(),
				itemFilter_InterfaceType);
		RequiredInterface recommendationItemFilter__itemFilter_ReqInt = getInterfaceOfType(
				recommendationItemFilter_Component.getRequiredInterfaces(),
				itemFilter_InterfaceType);

		Component futureSalesItemFilter_Component = futureSalesItemFilter_ComponentType
				.instantiate();
		components.add(futureSalesItemFilter_Component);
		ProvidedInterface futureSalesItemFilter__itemFilter_ProvInt = getInterfaceOfType(
				futureSalesItemFilter_Component.getProvidedInterfaces(),
				itemFilter_InterfaceType);
		RequiredInterface futureSalesItemFilter__itemFilter_ReqInt = getInterfaceOfType(
				futureSalesItemFilter_Component.getRequiredInterfaces(),
				itemFilter_InterfaceType);

		// create connectors
		connect(reputationService__basicQueryService_ReqInt,
				queryService__basicQueryService_ProvInt);
		connect(reputationService__queryService_ReqInt,
				queryService__queryService_ProvInt);
		connect(reputationService__authenticationService_ReqInt,
				authenticationService__authenticationService_ProvInt);
		connect(reputationService__businessObjectPersistenceService_ReqInt,
				persistenceService__businessObjectPersistenceService_ProvInt);

		connect(itemManagementService__businessObjectPersistenceService_ReqInt,
				persistenceService__businessObjectPersistenceService_ProvInt);
		connect(itemManagementService__authenticationService_ReqInt,
				authenticationService__authenticationService_ProvInt);
		connect(itemManagementService__queryService_ReqInt,
				queryService__queryService_ProvInt);
		connect(itemManagementService__basicQueryService_ReqInt,
				queryService__basicQueryService_ProvInt);

		connect(userManagementService__businessObjectPersistenceService_ReqInt,
				persistenceService__businessObjectPersistenceService_ProvInt);
		connect(userManagementService__authenticationService_ReqInt,
				authenticationService__authenticationService_ProvInt);
		connect(userManagementService__queryService_ReqInt,
				queryService__queryService_ProvInt);
		connect(userManagementService__basicQueryService_ReqInt,
				queryService__basicQueryService_ProvInt);

		connect(authenticationService__basicQueryService_ReqInt,
				queryService__basicQueryService_ProvInt);

		connect(bidAndBuyService__inventoryService_ReqInt,
				inventoryService__inventoryService_ProvInt);
		connect(bidAndBuyService__businessObjectPersistenceService_ReqInt,
				persistenceService__businessObjectPersistenceService_ProvInt);
		connect(bidAndBuyService__authenticationService_ReqInt,
				authenticationService__authenticationService_ProvInt);
		connect(bidAndBuyService__queryService_ReqInt,
				queryService__queryService_ProvInt);
		connect(bidAndBuyService__basicQueryService_ReqInt,
				queryService__basicQueryService_ProvInt);

		connect(inventoryService__queryService_ReqInt,
				queryService__queryService_ProvInt);
		connect(inventoryService__businessObjectPersistenceService_ReqInt,
				persistenceService__businessObjectPersistenceService_ProvInt);

		connect(queryService__itemFilter_ReqInt,
				lastSecondSalesItemFilter__itemFilter_ProvInt);

		connect(lastSecondSalesItemFilter__itemFilter_ReqInt,
				pastSalesItemFilter__itemFilter_ProvInt);

		connect(pastSalesItemFilter__itemFilter_ReqInt,
				buyNowItemFilter__itemFilter_ProvInt);

		connect(buyNowItemFilter__itemFilter_ReqInt,
				availabilityItemFilter__itemFilter_ProvInt);

		connect(availabilityItemFilter__itemFilter_ReqInt,
				regionItemFilter__itemFilter_ProvInt);

		connect(regionItemFilter__itemFilter_ReqInt,
				sellerReputationItemFilter__itemFilter_ProvInt);

		connect(sellerReputationItemFilter__itemFilter_ReqInt,
				commentItemFilter__itemFilter_ProvInt);

		connect(commentItemFilter__itemFilter_ReqInt,
				categoryItemFilter__itemFilter_ProvInt);

		connect(categoryItemFilter__itemFilter_ReqInt,
				recommendationItemFilter__itemFilter_ProvInt);

		connect(recommendationItemFilter__itemFilter_ReqInt,
				futureSalesItemFilter__itemFilter_ProvInt);

		// add components to the shop and start components
		for (Component component : components) {
			shop.getComponents().add(component);
			component.setState(ComponentLifeCycle.STARTED);
			//Server server = ComparchFactory.eINSTANCE.createServer();
			Replica replica =  ComparchFactory.eINSTANCE.createReplica();
			replica.setComponent(component);
		}

		// return the shop
		return shop;

	}

	private static <T extends Interface> T getInterfaceOfType(
			List<T> interfazes, InterfaceType interfaceType) {
		T result = null;
		for (T interfaze : interfazes) {
			if (interfaceType.equals(interfaze.getType())) {
				result = interfaze;
			}
		}
		return result;
	}

	private static void connect(RequiredInterface requiredInterface,
			ProvidedInterface providedInterface) {
		if (requiredInterface.getType().equals(providedInterface.getType())) {
			Connector connector = ComparchFactory.eINSTANCE.createConnector();
			connector.setLazyAdaptation(false);
			// connector.setUid("Connector_"
			// + requiredInterface.getType().getFullyQualifiedName()
			// + EcoreUtil.generateUUID());
			connector.setSource(requiredInterface);
			connector.setTarget(providedInterface);
		} else {
			throw new RuntimeException("Cannot connect " + requiredInterface
					+ " to " + providedInterface);
		}
	}

	
	
	
	/*
	 * ========================================================
	 */
	/*
	 * Statistics generation
	 */

	private static void attachStatistics(Tenant shop) {
		CompArchPerformanceDataGenerator generator = new CompArchPerformanceDataGenerator();

		// performance stats
		Map<String, Map<de.mdelab.morisia.mrubis.CompArchPerformanceDataGenerator.Method, MethodPerformance>> stats = generator
				.generatePerformanceStats();
		for (String componentTypeName : stats.keySet()) {
			Map<de.mdelab.morisia.mrubis.CompArchPerformanceDataGenerator.Method, MethodPerformance> cStats = stats
					.get(componentTypeName);
			Component eComponent = findComponentInShopByType(shop,
					componentTypeName);
			for (de.mdelab.morisia.mrubis.CompArchPerformanceDataGenerator.Method m : cStats
					.keySet()) {
				MethodPerformance performance = cStats.get(m);
				ProvidedInterface eProvidedInterface = findProvidedInerfaceOfComponentByType(
						eComponent, m.getFullyQualifiedInterface());
				MethodSpecification eMethodSpecification = findMethodSpecification(
						eProvidedInterface, m.getSignature());

				// add PerformanceStats element
				PerformanceStats ePerformanceStats = ComparchFactory.eINSTANCE
						.createPerformanceStats();
				//ePerformanceStats.setUid(EcoreUtil.generateUUID());
				ePerformanceStats.setMinTime(performance.getMinTime());
				ePerformanceStats.setMaxTime(performance.getMaxTime());
				ePerformanceStats.setTotalTime(performance.getTotalTime());
				ePerformanceStats.setInvocationCount(performance
						.getInvocationCount());
				ePerformanceStats.setMethod(eMethodSpecification);
				eProvidedInterface.getPerformanceStats().add(ePerformanceStats);
			}
		}

		// monitored property
		Map<String, List<Property>> propData = generator
				.generateMonitoredProperties();
		for (String componentTypeName : propData.keySet()) {
			List<Property> props = propData.get(componentTypeName);
			Component eComponent = findComponentInShopByType(shop,
					componentTypeName);
			for (Property prop : props) {
				MonitoredProperty eMonitoredProperty = ComparchFactory.eINSTANCE
						.createMonitoredProperty();
				//eMonitoredProperty.setUid(EcoreUtil.generateUUID());
				eMonitoredProperty.setName(prop.getName());
				eMonitoredProperty.setValue(prop.getValue());
				eMonitoredProperty.setType(prop.getType());
				eMonitoredProperty.setUnit(prop.getUnit());
				eMonitoredProperty.setDescription(prop.getDescription());
				eComponent.getMonitoredProperties().add(eMonitoredProperty);
			}
		}

	}

	private static Component findComponentInShopByType(Tenant shop,
			String componentTypeName) {
		Component result = null;
		for (Component c : shop.getComponents()) {
			if (c.getType().getName().equals(componentTypeName)) {
				result = c;
				break;
			}
		}
		return result;
	}

	private static ProvidedInterface findProvidedInerfaceOfComponentByType(
			Component c, String fullyQualifiedInterfaceName) {
		ProvidedInterface result = null;
		for (ProvidedInterface pi : c.getProvidedInterfaces()) {
			if (pi.getType().getFullyQualifiedName()
					.equals(fullyQualifiedInterfaceName)) {
				result = pi;
				break;
			}
		}
		return result;
	}

	private static MethodSpecification findMethodSpecification(
			ProvidedInterface pi, String signature) {
		MethodSpecification result = null;
		for (MethodSpecification m : pi.getType().getMethodSpecification()) {
			if (m.getSignature().equals(signature)) {
				result = m;
				break;
			}
		}
		return result;
	}

	/*
	 * ========================================================
	 */
	/*
	 * Util stuff
	 */

	private static void configure() {
		ComparchPackage.eINSTANCE.eClass();
		Resource.Factory.Registry reg = Resource.Factory.Registry.INSTANCE;
		Map<String, Object> m = reg.getExtensionToFactoryMap();
		m.put("comparch", new XMIResourceFactoryImpl());
	}

	private static void saveModelResource(Architecture architecture, String uri) {
		ResourceSet resourceSet = new ResourceSetImpl();
		Resource resource = resourceSet.createResource(URI.createURI(uri));
		resource.getContents().add(architecture);
		try {
			resource.save(Collections.EMPTY_MAP);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	private static Architecture loadModelFromXmi(String uri) {
		ResourceSet resourceSet = new ResourceSetImpl();
		Resource resource = resourceSet.getResource(URI.createURI(uri), true);

		Architecture architecture = (Architecture) resource.getContents()
				.get(0);
		return architecture;
	}

}
