/**
 */
package de.mdelab.morisia.comparch.impl;

import de.mdelab.morisia.comparch.*;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EDataType;
import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;

import org.eclipse.emf.ecore.impl.EFactoryImpl;

import org.eclipse.emf.ecore.plugin.EcorePlugin;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model <b>Factory</b>.
 * <!-- end-user-doc -->
 * @generated
 */
public class ComparchFactoryImpl extends EFactoryImpl implements ComparchFactory {
	/**
	 * Creates the default factory implementation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static ComparchFactory init() {
		try {
			ComparchFactory theComparchFactory = (ComparchFactory)EPackage.Registry.INSTANCE.getEFactory(ComparchPackage.eNS_URI);
			if (theComparchFactory != null) {
				return theComparchFactory;
			}
		}
		catch (Exception exception) {
			EcorePlugin.INSTANCE.log(exception);
		}
		return new ComparchFactoryImpl();
	}

	/**
	 * Creates an instance of the factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComparchFactoryImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public EObject create(EClass eClass) {
		switch (eClass.getClassifierID()) {
			case ComparchPackage.CLOUD: return createCloud();
			case ComparchPackage.SERVER: return createServer();
			case ComparchPackage.ARCHITECTURE: return createArchitecture();
			case ComparchPackage.COMPONENT_TYPE: return createComponentType();
			case ComparchPackage.INTERFACE_TYPE: return createInterfaceType();
			case ComparchPackage.PARAMETER_TYPE: return createParameterType();
			case ComparchPackage.COMPONENT: return createComponent();
			case ComparchPackage.REPLICA: return createReplica();
			case ComparchPackage.PARAMETER: return createParameter();
			case ComparchPackage.METHOD_SPECIFICATION: return createMethodSpecification();
			case ComparchPackage.REQUIRED_INTERFACE: return createRequiredInterface();
			case ComparchPackage.PROVIDED_INTERFACE: return createProvidedInterface();
			case ComparchPackage.CONNECTOR: return createConnector();
			case ComparchPackage.FAILURE: return createFailure();
			case ComparchPackage.PERFORMANCE_STATS: return createPerformanceStats();
			case ComparchPackage.MONITORED_PROPERTY: return createMonitoredProperty();
			case ComparchPackage.TENANT: return createTenant();
			case ComparchPackage.ANNOTATIONS: return createAnnotations();
			case ComparchPackage.CF1: return createCF1();
			case ComparchPackage.CF2: return createCF2();
			case ComparchPackage.CF3: return createCF3();
			case ComparchPackage.CF5: return createCF5();
			case ComparchPackage.MOVE_REPLICA: return createMoveReplica();
			case ComparchPackage.ADD_REPLICA: return createAddReplica();
			case ComparchPackage.REMOVE_REPLICA: return createRemoveReplica();
			case ComparchPackage.HW_REDEPLOY_COMPONENT: return createHwRedeployComponent();
			case ComparchPackage.LW_REDEPLOY_COMPONENT: return createLwRedeployComponent();
			case ComparchPackage.RESTART_COMPONENT: return createRestartComponent();
			case ComparchPackage.REPLACE_COMPONENT: return createReplaceComponent();
			case ComparchPackage.CF4: return createCF4();
			case ComparchPackage.RECREATE_CONNECTOR: return createRecreateConnector();
			default:
				throw new IllegalArgumentException("The class '" + eClass.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object createFromString(EDataType eDataType, String initialValue) {
		switch (eDataType.getClassifierID()) {
			case ComparchPackage.COMPONENT_LIFE_CYCLE:
				return createComponentLifeCycleFromString(eDataType, initialValue);
			default:
				throw new IllegalArgumentException("The datatype '" + eDataType.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String convertToString(EDataType eDataType, Object instanceValue) {
		switch (eDataType.getClassifierID()) {
			case ComparchPackage.COMPONENT_LIFE_CYCLE:
				return convertComponentLifeCycleToString(eDataType, instanceValue);
			default:
				throw new IllegalArgumentException("The datatype '" + eDataType.getName() + "' is not a valid classifier");
		}
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Architecture createArchitecture() {
		ArchitectureImpl architecture = new ArchitectureImpl();
		return architecture;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComponentType createComponentType() {
		ComponentTypeImpl componentType = new ComponentTypeImpl();
		return componentType;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public InterfaceType createInterfaceType() {
		InterfaceTypeImpl interfaceType = new InterfaceTypeImpl();
		return interfaceType;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ParameterType createParameterType() {
		ParameterTypeImpl parameterType = new ParameterTypeImpl();
		return parameterType;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Component createComponent() {
		ComponentImpl component = new ComponentImpl();
		return component;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Replica createReplica() {
		ReplicaImpl replica = new ReplicaImpl();
		return replica;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Parameter createParameter() {
		ParameterImpl parameter = new ParameterImpl();
		return parameter;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MethodSpecification createMethodSpecification() {
		MethodSpecificationImpl methodSpecification = new MethodSpecificationImpl();
		return methodSpecification;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public RequiredInterface createRequiredInterface() {
		RequiredInterfaceImpl requiredInterface = new RequiredInterfaceImpl();
		return requiredInterface;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProvidedInterface createProvidedInterface() {
		ProvidedInterfaceImpl providedInterface = new ProvidedInterfaceImpl();
		return providedInterface;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Connector createConnector() {
		ConnectorImpl connector = new ConnectorImpl();
		return connector;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Failure createFailure() {
		FailureImpl failure = new FailureImpl();
		return failure;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public PerformanceStats createPerformanceStats() {
		PerformanceStatsImpl performanceStats = new PerformanceStatsImpl();
		return performanceStats;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MonitoredProperty createMonitoredProperty() {
		MonitoredPropertyImpl monitoredProperty = new MonitoredPropertyImpl();
		return monitoredProperty;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Tenant createTenant() {
		TenantImpl tenant = new TenantImpl();
		return tenant;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Annotations createAnnotations() {
		AnnotationsImpl annotations = new AnnotationsImpl();
		return annotations;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public CF1 createCF1() {
		CF1Impl cf1 = new CF1Impl();
		return cf1;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public CF2 createCF2() {
		CF2Impl cf2 = new CF2Impl();
		return cf2;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public CF3 createCF3() {
		CF3Impl cf3 = new CF3Impl();
		return cf3;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public CF5 createCF5() {
		CF5Impl cf5 = new CF5Impl();
		return cf5;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MoveReplica createMoveReplica() {
		MoveReplicaImpl moveReplica = new MoveReplicaImpl();
		return moveReplica;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public AddReplica createAddReplica() {
		AddReplicaImpl addReplica = new AddReplicaImpl();
		return addReplica;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public RemoveReplica createRemoveReplica() {
		RemoveReplicaImpl removeReplica = new RemoveReplicaImpl();
		return removeReplica;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public HwRedeployComponent createHwRedeployComponent() {
		HwRedeployComponentImpl hwRedeployComponent = new HwRedeployComponentImpl();
		return hwRedeployComponent;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public LwRedeployComponent createLwRedeployComponent() {
		LwRedeployComponentImpl lwRedeployComponent = new LwRedeployComponentImpl();
		return lwRedeployComponent;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public RestartComponent createRestartComponent() {
		RestartComponentImpl restartComponent = new RestartComponentImpl();
		return restartComponent;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ReplaceComponent createReplaceComponent() {
		ReplaceComponentImpl replaceComponent = new ReplaceComponentImpl();
		return replaceComponent;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public CF4 createCF4() {
		CF4Impl cf4 = new CF4Impl();
		return cf4;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public RecreateConnector createRecreateConnector() {
		RecreateConnectorImpl recreateConnector = new RecreateConnectorImpl();
		return recreateConnector;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Server createServer() {
		ServerImpl server = new ServerImpl();
		return server;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Cloud createCloud() {
		CloudImpl cloud = new CloudImpl();
		return cloud;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComponentLifeCycle createComponentLifeCycleFromString(EDataType eDataType, String initialValue) {
		ComponentLifeCycle result = ComponentLifeCycle.get(initialValue);
		if (result == null) throw new IllegalArgumentException("The value '" + initialValue + "' is not a valid enumerator of '" + eDataType.getName() + "'");
		return result;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String convertComponentLifeCycleToString(EDataType eDataType, Object instanceValue) {
		return instanceValue == null ? null : instanceValue.toString();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComparchPackage getComparchPackage() {
		return (ComparchPackage)getEPackage();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @deprecated
	 * @generated
	 */
	@Deprecated
	public static ComparchPackage getPackage() {
		return ComparchPackage.eINSTANCE;
	}

} //ComparchFactoryImpl
