/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.ecore.EAttribute;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.EEnum;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.EReference;

/**
 * <!-- begin-user-doc -->
 * The <b>Package</b> for the model.
 * It contains accessors for the meta objects to represent
 * <ul>
 *   <li>each class,</li>
 *   <li>each feature of each class,</li>
 *   <li>each enum,</li>
 *   <li>and each data type</li>
 * </ul>
 * <!-- end-user-doc -->
 * @see de.mdelab.morisia.comparch.ComparchFactory
 * @model kind="package"
 * @generated
 */
public interface ComparchPackage extends EPackage {
	/**
	 * The package name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNAME = "comparch";

	/**
	 * The package namespace URI.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_URI = "http://www.mdelab.de/morisia/comparch";

	/**
	 * The package namespace name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	String eNS_PREFIX = "comparch";

	/**
	 * The singleton instance of the package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	ComparchPackage eINSTANCE = de.mdelab.morisia.comparch.impl.ComparchPackageImpl.init();

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.ArchitecturalElementImpl <em>Architectural Element</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.ArchitecturalElementImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getArchitecturalElement()
	 * @generated
	 */
	int ARCHITECTURAL_ELEMENT = 3;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.ArchitectureImpl <em>Architecture</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.ArchitectureImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getArchitecture()
	 * @generated
	 */
	int ARCHITECTURE = 2;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl <em>Component Type</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.ComponentTypeImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getComponentType()
	 * @generated
	 */
	int COMPONENT_TYPE = 4;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.InterfaceTypeImpl <em>Interface Type</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.InterfaceTypeImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getInterfaceType()
	 * @generated
	 */
	int INTERFACE_TYPE = 5;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.ParameterTypeImpl <em>Parameter Type</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.ParameterTypeImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getParameterType()
	 * @generated
	 */
	int PARAMETER_TYPE = 6;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.ComponentImpl <em>Component</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.ComponentImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getComponent()
	 * @generated
	 */
	int COMPONENT = 7;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.ReplicaImpl <em>Replica</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.ReplicaImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getReplica()
	 * @generated
	 */
	int REPLICA = 8;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.ParameterImpl <em>Parameter</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.ParameterImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getParameter()
	 * @generated
	 */
	int PARAMETER = 9;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.MethodSpecificationImpl <em>Method Specification</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.MethodSpecificationImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getMethodSpecification()
	 * @generated
	 */
	int METHOD_SPECIFICATION = 10;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.InterfaceImpl <em>Interface</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.InterfaceImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getInterface()
	 * @generated
	 */
	int INTERFACE = 11;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.RequiredInterfaceImpl <em>Required Interface</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.RequiredInterfaceImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getRequiredInterface()
	 * @generated
	 */
	int REQUIRED_INTERFACE = 12;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.ProvidedInterfaceImpl <em>Provided Interface</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.ProvidedInterfaceImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getProvidedInterface()
	 * @generated
	 */
	int PROVIDED_INTERFACE = 13;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.ConnectorImpl <em>Connector</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.ConnectorImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getConnector()
	 * @generated
	 */
	int CONNECTOR = 14;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.FailureImpl <em>Failure</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.FailureImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getFailure()
	 * @generated
	 */
	int FAILURE = 15;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.PerformanceStatsImpl <em>Performance Stats</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.PerformanceStatsImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getPerformanceStats()
	 * @generated
	 */
	int PERFORMANCE_STATS = 16;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.MonitoredPropertyImpl <em>Monitored Property</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.MonitoredPropertyImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getMonitoredProperty()
	 * @generated
	 */
	int MONITORED_PROPERTY = 17;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.TenantImpl <em>Tenant</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.TenantImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getTenant()
	 * @generated
	 */
	int TENANT = 18;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.AnnotationsImpl <em>Annotations</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.AnnotationsImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getAnnotations()
	 * @generated
	 */
	int ANNOTATIONS = 19;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.IssueImpl <em>Issue</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.IssueImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getIssue()
	 * @generated
	 */
	int ISSUE = 20;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.CFImpl <em>CF</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.CFImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCF()
	 * @generated
	 */
	int CF = 21;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.CF1Impl <em>CF1</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.CF1Impl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCF1()
	 * @generated
	 */
	int CF1 = 22;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.CF2Impl <em>CF2</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.CF2Impl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCF2()
	 * @generated
	 */
	int CF2 = 23;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.CF3Impl <em>CF3</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.CF3Impl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCF3()
	 * @generated
	 */
	int CF3 = 24;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.CF5Impl <em>CF5</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.CF5Impl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCF5()
	 * @generated
	 */
	int CF5 = 25;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.RuleImpl <em>Rule</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.RuleImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getRule()
	 * @generated
	 */
	int RULE = 26;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.MoveReplicaImpl <em>Move Replica</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.MoveReplicaImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getMoveReplica()
	 * @generated
	 */
	int MOVE_REPLICA = 27;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.AddReplicaImpl <em>Add Replica</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.AddReplicaImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getAddReplica()
	 * @generated
	 */
	int ADD_REPLICA = 28;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.RemoveReplicaImpl <em>Remove Replica</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.RemoveReplicaImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getRemoveReplica()
	 * @generated
	 */
	int REMOVE_REPLICA = 29;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.HwRedeployComponentImpl <em>Hw Redeploy Component</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.HwRedeployComponentImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getHwRedeployComponent()
	 * @generated
	 */
	int HW_REDEPLOY_COMPONENT = 30;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.LwRedeployComponentImpl <em>Lw Redeploy Component</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.LwRedeployComponentImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getLwRedeployComponent()
	 * @generated
	 */
	int LW_REDEPLOY_COMPONENT = 31;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.RestartComponentImpl <em>Restart Component</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.RestartComponentImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getRestartComponent()
	 * @generated
	 */
	int RESTART_COMPONENT = 32;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.ReplaceComponentImpl <em>Replace Component</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.ReplaceComponentImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getReplaceComponent()
	 * @generated
	 */
	int REPLACE_COMPONENT = 33;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.CF4Impl <em>CF4</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.CF4Impl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCF4()
	 * @generated
	 */
	int CF4 = 34;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.RecreateConnectorImpl <em>Recreate Connector</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.RecreateConnectorImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getRecreateConnector()
	 * @generated
	 */
	int RECREATE_CONNECTOR = 35;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.CloudImpl <em>Cloud</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.CloudImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCloud()
	 * @generated
	 */
	int CLOUD = 0;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.impl.ServerImpl <em>Server</em>}' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.impl.ServerImpl
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getServer()
	 * @generated
	 */
	int SERVER = 1;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURAL_ELEMENT__UID = 0;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURAL_ELEMENT__CRITICALITY = 1;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURAL_ELEMENT__IMPORTANCE = 2;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURAL_ELEMENT__ADT = 3;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURAL_ELEMENT__EFFICIENCY = 4;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURAL_ELEMENT__PERFORMANCE = 5;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURAL_ELEMENT__REQUEST = 6;

	/**
	 * The number of structural features of the '<em>Architectural Element</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURAL_ELEMENT_FEATURE_COUNT = 7;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLOUD__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLOUD__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLOUD__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLOUD__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLOUD__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLOUD__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLOUD__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Servers</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLOUD__SERVERS = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Architecture</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLOUD__ARCHITECTURE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLOUD__NAME = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Cloud</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CLOUD_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Cloud</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__CLOUD = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__NAME = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Deployed Replicas</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__DEPLOYED_REPLICAS = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Deployed Component</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__DEPLOYED_COMPONENT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>CPU Capacity</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__CPU_CAPACITY = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Memory Capacity</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__MEMORY_CAPACITY = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 5;

	/**
	 * The feature id for the '<em><b>Storage Capacity</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__STORAGE_CAPACITY = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 6;

	/**
	 * The feature id for the '<em><b>CPU Consumption</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__CPU_CONSUMPTION = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 7;

	/**
	 * The feature id for the '<em><b>Memory Consumption</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__MEMORY_CONSUMPTION = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 8;

	/**
	 * The feature id for the '<em><b>Storage Consumption</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__STORAGE_CONSUMPTION = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 9;

	/**
	 * The feature id for the '<em><b>Cost</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__COST = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 10;

	/**
	 * The feature id for the '<em><b>Cpu Factor</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__CPU_FACTOR = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 11;

	/**
	 * The feature id for the '<em><b>Memory Factor</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__MEMORY_FACTOR = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 12;

	/**
	 * The feature id for the '<em><b>Storage Factor</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER__STORAGE_FACTOR = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 13;

	/**
	 * The number of structural features of the '<em>Server</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int SERVER_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 14;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__NAME = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Component Types</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__COMPONENT_TYPES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Interface Types</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__INTERFACE_TYPES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Tenants</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__TENANTS = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Cloud</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__CLOUD = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE__ANNOTATIONS = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 5;

	/**
	 * The number of structural features of the '<em>Architecture</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ARCHITECTURE_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 6;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__NAME = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Architecture</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__ARCHITECTURE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Required Interface Types</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__REQUIRED_INTERFACE_TYPES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Provided Interface Types</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__PROVIDED_INTERFACE_TYPES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Parameter Types</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__PARAMETER_TYPES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Instances</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__INSTANCES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 5;

	/**
	 * The feature id for the '<em><b>Reliability</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__RELIABILITY = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 6;

	/**
	 * The feature id for the '<em><b>Sat Point</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__SAT_POINT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 7;

	/**
	 * The feature id for the '<em><b>Performance Max</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__PERFORMANCE_MAX = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 8;

	/**
	 * The feature id for the '<em><b>Replica Cost</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__REPLICA_COST = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 9;

	/**
	 * The feature id for the '<em><b>Replica ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__REPLICA_ADT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 10;

	/**
	 * The feature id for the '<em><b>Cpu Factor</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__CPU_FACTOR = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 11;

	/**
	 * The feature id for the '<em><b>Memory Factor</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__MEMORY_FACTOR = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 12;

	/**
	 * The feature id for the '<em><b>Storage Factor</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE__STORAGE_FACTOR = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 13;

	/**
	 * The number of structural features of the '<em>Component Type</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_TYPE_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 14;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Fully Qualified Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__FULLY_QUALIFIED_NAME = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Architecture</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__ARCHITECTURE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Requiring Component Types</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__REQUIRING_COMPONENT_TYPES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Providing Component Types</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__PROVIDING_COMPONENT_TYPES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Method Specification</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__METHOD_SPECIFICATION = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Instances</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE__INSTANCES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 5;

	/**
	 * The number of structural features of the '<em>Interface Type</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_TYPE_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 6;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE__NAME = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Type</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE__TYPE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Default Value</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE__DEFAULT_VALUE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Component Type</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE__COMPONENT_TYPE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Instances</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE__INSTANCES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 4;

	/**
	 * The number of structural features of the '<em>Parameter Type</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_TYPE_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 5;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Parameters</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__PARAMETERS = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Required Interfaces</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__REQUIRED_INTERFACES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>State</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__STATE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Provided Interfaces</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__PROVIDED_INTERFACES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Type</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__TYPE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Monitored Properties</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__MONITORED_PROPERTIES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 5;

	/**
	 * The feature id for the '<em><b>Tenant</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__TENANT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 6;

	/**
	 * The feature id for the '<em><b>Issues</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__ISSUES = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 7;

	/**
	 * The feature id for the '<em><b>In Use Replica</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__IN_USE_REPLICA = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 8;

	/**
	 * The feature id for the '<em><b>Health Status</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__HEALTH_STATUS = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 9;

	/**
	 * The feature id for the '<em><b>Performance Old</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__PERFORMANCE_OLD = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 10;

	/**
	 * The feature id for the '<em><b>Original Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__ORIGINAL_REQUEST = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 11;

	/**
	 * The feature id for the '<em><b>Deployed On</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__DEPLOYED_ON = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 12;

	/**
	 * The feature id for the '<em><b>Replica</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT__REPLICA = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 13;

	/**
	 * The number of structural features of the '<em>Component</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int COMPONENT_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 14;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Deployed On</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__DEPLOYED_ON = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Cost</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__COST = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>In Use</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__IN_USE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>CPU Requirement</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__CPU_REQUIREMENT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Memory Requirement</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__MEMORY_REQUIREMENT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Storage Requirement</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__STORAGE_REQUIREMENT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 5;

	/**
	 * The feature id for the '<em><b>Component</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA__COMPONENT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 6;

	/**
	 * The number of structural features of the '<em>Replica</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLICA_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 7;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Value</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__VALUE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Lazy Adaptation</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__LAZY_ADAPTATION = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Component</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__COMPONENT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Type</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER__TYPE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The number of structural features of the '<em>Parameter</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PARAMETER_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int METHOD_SPECIFICATION__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int METHOD_SPECIFICATION__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int METHOD_SPECIFICATION__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int METHOD_SPECIFICATION__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int METHOD_SPECIFICATION__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int METHOD_SPECIFICATION__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int METHOD_SPECIFICATION__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Signature</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int METHOD_SPECIFICATION__SIGNATURE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Interface Type</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int METHOD_SPECIFICATION__INTERFACE_TYPE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Method Specification</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int METHOD_SPECIFICATION_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Type</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE__TYPE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Interface</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int INTERFACE_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REQUIRED_INTERFACE__UID = INTERFACE__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REQUIRED_INTERFACE__CRITICALITY = INTERFACE__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REQUIRED_INTERFACE__IMPORTANCE = INTERFACE__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REQUIRED_INTERFACE__ADT = INTERFACE__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REQUIRED_INTERFACE__EFFICIENCY = INTERFACE__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REQUIRED_INTERFACE__PERFORMANCE = INTERFACE__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REQUIRED_INTERFACE__REQUEST = INTERFACE__REQUEST;

	/**
	 * The feature id for the '<em><b>Type</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REQUIRED_INTERFACE__TYPE = INTERFACE__TYPE;

	/**
	 * The feature id for the '<em><b>Component</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REQUIRED_INTERFACE__COMPONENT = INTERFACE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Connector</b></em>' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REQUIRED_INTERFACE__CONNECTOR = INTERFACE_FEATURE_COUNT + 1;

	/**
	 * The number of structural features of the '<em>Required Interface</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REQUIRED_INTERFACE_FEATURE_COUNT = INTERFACE_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE__UID = INTERFACE__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE__CRITICALITY = INTERFACE__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE__IMPORTANCE = INTERFACE__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE__ADT = INTERFACE__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE__EFFICIENCY = INTERFACE__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE__PERFORMANCE = INTERFACE__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE__REQUEST = INTERFACE__REQUEST;

	/**
	 * The feature id for the '<em><b>Type</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE__TYPE = INTERFACE__TYPE;

	/**
	 * The feature id for the '<em><b>Component</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE__COMPONENT = INTERFACE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Connectors</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE__CONNECTORS = INTERFACE_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Failures</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE__FAILURES = INTERFACE_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Performance Stats</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE__PERFORMANCE_STATS = INTERFACE_FEATURE_COUNT + 3;

	/**
	 * The number of structural features of the '<em>Provided Interface</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PROVIDED_INTERFACE_FEATURE_COUNT = INTERFACE_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONNECTOR__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONNECTOR__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONNECTOR__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONNECTOR__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONNECTOR__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONNECTOR__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONNECTOR__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Lazy Adaptation</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONNECTOR__LAZY_ADAPTATION = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Source</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONNECTOR__SOURCE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Target</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONNECTOR__TARGET = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The number of structural features of the '<em>Connector</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CONNECTOR_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAILURE__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAILURE__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAILURE__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAILURE__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAILURE__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAILURE__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAILURE__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAILURE__NAME = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Message</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAILURE__MESSAGE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Interface</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAILURE__INTERFACE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Method</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAILURE__METHOD = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The number of structural features of the '<em>Failure</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int FAILURE_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Min Time</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__MIN_TIME = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Max Time</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__MAX_TIME = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Total Time</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__TOTAL_TIME = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Invocation Count</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__INVOCATION_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Interface</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__INTERFACE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Method</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS__METHOD = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 5;

	/**
	 * The number of structural features of the '<em>Performance Stats</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int PERFORMANCE_STATS_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 6;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__NAME = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Value</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__VALUE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Type</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__TYPE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Unit</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__UNIT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The feature id for the '<em><b>Description</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__DESCRIPTION = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Component</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__COMPONENT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 5;

	/**
	 * The feature id for the '<em><b>Concerned Elements</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY__CONCERNED_ELEMENTS = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 6;

	/**
	 * The number of structural features of the '<em>Monitored Property</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MONITORED_PROPERTY_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 7;

	/**
	 * The feature id for the '<em><b>Uid</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TENANT__UID = ARCHITECTURAL_ELEMENT__UID;

	/**
	 * The feature id for the '<em><b>Criticality</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TENANT__CRITICALITY = ARCHITECTURAL_ELEMENT__CRITICALITY;

	/**
	 * The feature id for the '<em><b>Importance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TENANT__IMPORTANCE = ARCHITECTURAL_ELEMENT__IMPORTANCE;

	/**
	 * The feature id for the '<em><b>ADT</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TENANT__ADT = ARCHITECTURAL_ELEMENT__ADT;

	/**
	 * The feature id for the '<em><b>Efficiency</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TENANT__EFFICIENCY = ARCHITECTURAL_ELEMENT__EFFICIENCY;

	/**
	 * The feature id for the '<em><b>Performance</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TENANT__PERFORMANCE = ARCHITECTURAL_ELEMENT__PERFORMANCE;

	/**
	 * The feature id for the '<em><b>Request</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TENANT__REQUEST = ARCHITECTURAL_ELEMENT__REQUEST;

	/**
	 * The feature id for the '<em><b>Name</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TENANT__NAME = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Architecture</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TENANT__ARCHITECTURE = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Components</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TENANT__COMPONENTS = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 2;

	/**
	 * The feature id for the '<em><b>Replica Factor</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TENANT__REPLICA_FACTOR = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 3;

	/**
	 * The number of structural features of the '<em>Tenant</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int TENANT_FEATURE_COUNT = ARCHITECTURAL_ELEMENT_FEATURE_COUNT + 4;

	/**
	 * The feature id for the '<em><b>Architecture</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANNOTATIONS__ARCHITECTURE = 0;

	/**
	 * The feature id for the '<em><b>Issues</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANNOTATIONS__ISSUES = 1;

	/**
	 * The feature id for the '<em><b>Rules</b></em>' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANNOTATIONS__RULES = 2;

	/**
	 * The number of structural features of the '<em>Annotations</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ANNOTATIONS_FEATURE_COUNT = 3;

	/**
	 * The feature id for the '<em><b>Utility Drop</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ISSUE__UTILITY_DROP = 0;

	/**
	 * The feature id for the '<em><b>Affected Component</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ISSUE__AFFECTED_COMPONENT = 1;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ISSUE__ANNOTATIONS = 2;

	/**
	 * The feature id for the '<em><b>Handled By</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ISSUE__HANDLED_BY = 3;

	/**
	 * The number of structural features of the '<em>Issue</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ISSUE_FEATURE_COUNT = 4;

	/**
	 * The feature id for the '<em><b>Utility Drop</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF__UTILITY_DROP = ISSUE__UTILITY_DROP;

	/**
	 * The feature id for the '<em><b>Affected Component</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF__AFFECTED_COMPONENT = ISSUE__AFFECTED_COMPONENT;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF__ANNOTATIONS = ISSUE__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handled By</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF__HANDLED_BY = ISSUE__HANDLED_BY;

	/**
	 * The number of structural features of the '<em>CF</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF_FEATURE_COUNT = ISSUE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Utility Drop</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF1__UTILITY_DROP = CF__UTILITY_DROP;

	/**
	 * The feature id for the '<em><b>Affected Component</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF1__AFFECTED_COMPONENT = CF__AFFECTED_COMPONENT;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF1__ANNOTATIONS = CF__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handled By</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF1__HANDLED_BY = CF__HANDLED_BY;

	/**
	 * The number of structural features of the '<em>CF1</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF1_FEATURE_COUNT = CF_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Utility Drop</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF2__UTILITY_DROP = CF__UTILITY_DROP;

	/**
	 * The feature id for the '<em><b>Affected Component</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF2__AFFECTED_COMPONENT = CF__AFFECTED_COMPONENT;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF2__ANNOTATIONS = CF__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handled By</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF2__HANDLED_BY = CF__HANDLED_BY;

	/**
	 * The number of structural features of the '<em>CF2</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF2_FEATURE_COUNT = CF_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Utility Drop</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF3__UTILITY_DROP = CF__UTILITY_DROP;

	/**
	 * The feature id for the '<em><b>Affected Component</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF3__AFFECTED_COMPONENT = CF__AFFECTED_COMPONENT;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF3__ANNOTATIONS = CF__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handled By</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF3__HANDLED_BY = CF__HANDLED_BY;

	/**
	 * The number of structural features of the '<em>CF3</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF3_FEATURE_COUNT = CF_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Utility Drop</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF5__UTILITY_DROP = CF__UTILITY_DROP;

	/**
	 * The feature id for the '<em><b>Affected Component</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF5__AFFECTED_COMPONENT = CF__AFFECTED_COMPONENT;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF5__ANNOTATIONS = CF__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handled By</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF5__HANDLED_BY = CF__HANDLED_BY;

	/**
	 * The number of structural features of the '<em>CF5</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF5_FEATURE_COUNT = CF_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Costs</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RULE__COSTS = 0;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RULE__ANNOTATIONS = 1;

	/**
	 * The feature id for the '<em><b>Handles</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RULE__HANDLES = 2;

	/**
	 * The feature id for the '<em><b>Utility Increase</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RULE__UTILITY_INCREASE = 3;

	/**
	 * The number of structural features of the '<em>Rule</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RULE_FEATURE_COUNT = 4;

	/**
	 * The feature id for the '<em><b>Costs</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MOVE_REPLICA__COSTS = RULE__COSTS;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MOVE_REPLICA__ANNOTATIONS = RULE__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handles</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MOVE_REPLICA__HANDLES = RULE__HANDLES;

	/**
	 * The feature id for the '<em><b>Utility Increase</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MOVE_REPLICA__UTILITY_INCREASE = RULE__UTILITY_INCREASE;

	/**
	 * The number of structural features of the '<em>Move Replica</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int MOVE_REPLICA_FEATURE_COUNT = RULE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Costs</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ADD_REPLICA__COSTS = RULE__COSTS;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ADD_REPLICA__ANNOTATIONS = RULE__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handles</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ADD_REPLICA__HANDLES = RULE__HANDLES;

	/**
	 * The feature id for the '<em><b>Utility Increase</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ADD_REPLICA__UTILITY_INCREASE = RULE__UTILITY_INCREASE;

	/**
	 * The number of structural features of the '<em>Add Replica</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int ADD_REPLICA_FEATURE_COUNT = RULE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Costs</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REMOVE_REPLICA__COSTS = RULE__COSTS;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REMOVE_REPLICA__ANNOTATIONS = RULE__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handles</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REMOVE_REPLICA__HANDLES = RULE__HANDLES;

	/**
	 * The feature id for the '<em><b>Utility Increase</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REMOVE_REPLICA__UTILITY_INCREASE = RULE__UTILITY_INCREASE;

	/**
	 * The number of structural features of the '<em>Remove Replica</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REMOVE_REPLICA_FEATURE_COUNT = RULE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Costs</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int HW_REDEPLOY_COMPONENT__COSTS = RULE__COSTS;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int HW_REDEPLOY_COMPONENT__ANNOTATIONS = RULE__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handles</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int HW_REDEPLOY_COMPONENT__HANDLES = RULE__HANDLES;

	/**
	 * The feature id for the '<em><b>Utility Increase</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int HW_REDEPLOY_COMPONENT__UTILITY_INCREASE = RULE__UTILITY_INCREASE;

	/**
	 * The number of structural features of the '<em>Hw Redeploy Component</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int HW_REDEPLOY_COMPONENT_FEATURE_COUNT = RULE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Costs</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LW_REDEPLOY_COMPONENT__COSTS = RULE__COSTS;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LW_REDEPLOY_COMPONENT__ANNOTATIONS = RULE__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handles</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LW_REDEPLOY_COMPONENT__HANDLES = RULE__HANDLES;

	/**
	 * The feature id for the '<em><b>Utility Increase</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LW_REDEPLOY_COMPONENT__UTILITY_INCREASE = RULE__UTILITY_INCREASE;

	/**
	 * The number of structural features of the '<em>Lw Redeploy Component</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int LW_REDEPLOY_COMPONENT_FEATURE_COUNT = RULE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Costs</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RESTART_COMPONENT__COSTS = RULE__COSTS;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RESTART_COMPONENT__ANNOTATIONS = RULE__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handles</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RESTART_COMPONENT__HANDLES = RULE__HANDLES;

	/**
	 * The feature id for the '<em><b>Utility Increase</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RESTART_COMPONENT__UTILITY_INCREASE = RULE__UTILITY_INCREASE;

	/**
	 * The number of structural features of the '<em>Restart Component</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RESTART_COMPONENT_FEATURE_COUNT = RULE_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Costs</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLACE_COMPONENT__COSTS = RULE__COSTS;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLACE_COMPONENT__ANNOTATIONS = RULE__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handles</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLACE_COMPONENT__HANDLES = RULE__HANDLES;

	/**
	 * The feature id for the '<em><b>Utility Increase</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLACE_COMPONENT__UTILITY_INCREASE = RULE__UTILITY_INCREASE;

	/**
	 * The feature id for the '<em><b>Alternative Component Type</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLACE_COMPONENT__ALTERNATIVE_COMPONENT_TYPE = RULE_FEATURE_COUNT + 0;

	/**
	 * The number of structural features of the '<em>Replace Component</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int REPLACE_COMPONENT_FEATURE_COUNT = RULE_FEATURE_COUNT + 1;

	/**
	 * The feature id for the '<em><b>Utility Drop</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF4__UTILITY_DROP = CF__UTILITY_DROP;

	/**
	 * The feature id for the '<em><b>Affected Component</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF4__AFFECTED_COMPONENT = CF__AFFECTED_COMPONENT;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF4__ANNOTATIONS = CF__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handled By</b></em>' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF4__HANDLED_BY = CF__HANDLED_BY;

	/**
	 * The number of structural features of the '<em>CF4</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int CF4_FEATURE_COUNT = CF_FEATURE_COUNT + 0;

	/**
	 * The feature id for the '<em><b>Costs</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECREATE_CONNECTOR__COSTS = RULE__COSTS;

	/**
	 * The feature id for the '<em><b>Annotations</b></em>' container reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECREATE_CONNECTOR__ANNOTATIONS = RULE__ANNOTATIONS;

	/**
	 * The feature id for the '<em><b>Handles</b></em>' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECREATE_CONNECTOR__HANDLES = RULE__HANDLES;

	/**
	 * The feature id for the '<em><b>Utility Increase</b></em>' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECREATE_CONNECTOR__UTILITY_INCREASE = RULE__UTILITY_INCREASE;

	/**
	 * The number of structural features of the '<em>Recreate Connector</em>' class.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 * @ordered
	 */
	int RECREATE_CONNECTOR_FEATURE_COUNT = RULE_FEATURE_COUNT + 0;

	/**
	 * The meta object id for the '{@link de.mdelab.morisia.comparch.ComponentLifeCycle <em>Component Life Cycle</em>}' enum.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see de.mdelab.morisia.comparch.ComponentLifeCycle
	 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getComponentLifeCycle()
	 * @generated
	 */
	int COMPONENT_LIFE_CYCLE = 36;


	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Architecture <em>Architecture</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Architecture</em>'.
	 * @see de.mdelab.morisia.comparch.Architecture
	 * @generated
	 */
	EClass getArchitecture();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Architecture#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see de.mdelab.morisia.comparch.Architecture#getName()
	 * @see #getArchitecture()
	 * @generated
	 */
	EAttribute getArchitecture_Name();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.Architecture#getComponentTypes <em>Component Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Component Types</em>'.
	 * @see de.mdelab.morisia.comparch.Architecture#getComponentTypes()
	 * @see #getArchitecture()
	 * @generated
	 */
	EReference getArchitecture_ComponentTypes();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.Architecture#getInterfaceTypes <em>Interface Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Interface Types</em>'.
	 * @see de.mdelab.morisia.comparch.Architecture#getInterfaceTypes()
	 * @see #getArchitecture()
	 * @generated
	 */
	EReference getArchitecture_InterfaceTypes();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.Architecture#getTenants <em>Tenants</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Tenants</em>'.
	 * @see de.mdelab.morisia.comparch.Architecture#getTenants()
	 * @see #getArchitecture()
	 * @generated
	 */
	EReference getArchitecture_Tenants();

	/**
	 * Returns the meta object for the containment reference '{@link de.mdelab.morisia.comparch.Architecture#getCloud <em>Cloud</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Cloud</em>'.
	 * @see de.mdelab.morisia.comparch.Architecture#getCloud()
	 * @see #getArchitecture()
	 * @generated
	 */
	EReference getArchitecture_Cloud();

	/**
	 * Returns the meta object for the containment reference '{@link de.mdelab.morisia.comparch.Architecture#getAnnotations <em>Annotations</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Annotations</em>'.
	 * @see de.mdelab.morisia.comparch.Architecture#getAnnotations()
	 * @see #getArchitecture()
	 * @generated
	 */
	EReference getArchitecture_Annotations();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.ArchitecturalElement <em>Architectural Element</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Architectural Element</em>'.
	 * @see de.mdelab.morisia.comparch.ArchitecturalElement
	 * @generated
	 */
	EClass getArchitecturalElement();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getUid <em>Uid</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Uid</em>'.
	 * @see de.mdelab.morisia.comparch.ArchitecturalElement#getUid()
	 * @see #getArchitecturalElement()
	 * @generated
	 */
	EAttribute getArchitecturalElement_Uid();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getCriticality <em>Criticality</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Criticality</em>'.
	 * @see de.mdelab.morisia.comparch.ArchitecturalElement#getCriticality()
	 * @see #getArchitecturalElement()
	 * @generated
	 */
	EAttribute getArchitecturalElement_Criticality();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getImportance <em>Importance</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Importance</em>'.
	 * @see de.mdelab.morisia.comparch.ArchitecturalElement#getImportance()
	 * @see #getArchitecturalElement()
	 * @generated
	 */
	EAttribute getArchitecturalElement_Importance();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getADT <em>ADT</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>ADT</em>'.
	 * @see de.mdelab.morisia.comparch.ArchitecturalElement#getADT()
	 * @see #getArchitecturalElement()
	 * @generated
	 */
	EAttribute getArchitecturalElement_ADT();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getEfficiency <em>Efficiency</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Efficiency</em>'.
	 * @see de.mdelab.morisia.comparch.ArchitecturalElement#getEfficiency()
	 * @see #getArchitecturalElement()
	 * @generated
	 */
	EAttribute getArchitecturalElement_Efficiency();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getPerformance <em>Performance</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Performance</em>'.
	 * @see de.mdelab.morisia.comparch.ArchitecturalElement#getPerformance()
	 * @see #getArchitecturalElement()
	 * @generated
	 */
	EAttribute getArchitecturalElement_Performance();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ArchitecturalElement#getRequest <em>Request</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Request</em>'.
	 * @see de.mdelab.morisia.comparch.ArchitecturalElement#getRequest()
	 * @see #getArchitecturalElement()
	 * @generated
	 */
	EAttribute getArchitecturalElement_Request();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.ComponentType <em>Component Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Component Type</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType
	 * @generated
	 */
	EClass getComponentType();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ComponentType#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getName()
	 * @see #getComponentType()
	 * @generated
	 */
	EAttribute getComponentType_Name();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.ComponentType#getArchitecture <em>Architecture</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Architecture</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getArchitecture()
	 * @see #getComponentType()
	 * @generated
	 */
	EReference getComponentType_Architecture();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.ComponentType#getRequiredInterfaceTypes <em>Required Interface Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Required Interface Types</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getRequiredInterfaceTypes()
	 * @see #getComponentType()
	 * @generated
	 */
	EReference getComponentType_RequiredInterfaceTypes();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.ComponentType#getProvidedInterfaceTypes <em>Provided Interface Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Provided Interface Types</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getProvidedInterfaceTypes()
	 * @see #getComponentType()
	 * @generated
	 */
	EReference getComponentType_ProvidedInterfaceTypes();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.ComponentType#getParameterTypes <em>Parameter Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Parameter Types</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getParameterTypes()
	 * @see #getComponentType()
	 * @generated
	 */
	EReference getComponentType_ParameterTypes();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.ComponentType#getInstances <em>Instances</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Instances</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getInstances()
	 * @see #getComponentType()
	 * @generated
	 */
	EReference getComponentType_Instances();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ComponentType#getReliability <em>Reliability</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Reliability</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getReliability()
	 * @see #getComponentType()
	 * @generated
	 */
	EAttribute getComponentType_Reliability();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ComponentType#getSatPoint <em>Sat Point</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Sat Point</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getSatPoint()
	 * @see #getComponentType()
	 * @generated
	 */
	EAttribute getComponentType_SatPoint();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ComponentType#getPerformanceMax <em>Performance Max</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Performance Max</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getPerformanceMax()
	 * @see #getComponentType()
	 * @generated
	 */
	EAttribute getComponentType_PerformanceMax();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ComponentType#getReplicaCost <em>Replica Cost</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Replica Cost</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getReplicaCost()
	 * @see #getComponentType()
	 * @generated
	 */
	EAttribute getComponentType_ReplicaCost();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ComponentType#getReplicaADT <em>Replica ADT</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Replica ADT</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getReplicaADT()
	 * @see #getComponentType()
	 * @generated
	 */
	EAttribute getComponentType_ReplicaADT();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ComponentType#getCpuFactor <em>Cpu Factor</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Cpu Factor</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getCpuFactor()
	 * @see #getComponentType()
	 * @generated
	 */
	EAttribute getComponentType_CpuFactor();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ComponentType#getMemoryFactor <em>Memory Factor</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Memory Factor</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getMemoryFactor()
	 * @see #getComponentType()
	 * @generated
	 */
	EAttribute getComponentType_MemoryFactor();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ComponentType#getStorageFactor <em>Storage Factor</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Storage Factor</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentType#getStorageFactor()
	 * @see #getComponentType()
	 * @generated
	 */
	EAttribute getComponentType_StorageFactor();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.InterfaceType <em>Interface Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Interface Type</em>'.
	 * @see de.mdelab.morisia.comparch.InterfaceType
	 * @generated
	 */
	EClass getInterfaceType();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.InterfaceType#getFullyQualifiedName <em>Fully Qualified Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Fully Qualified Name</em>'.
	 * @see de.mdelab.morisia.comparch.InterfaceType#getFullyQualifiedName()
	 * @see #getInterfaceType()
	 * @generated
	 */
	EAttribute getInterfaceType_FullyQualifiedName();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.InterfaceType#getArchitecture <em>Architecture</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Architecture</em>'.
	 * @see de.mdelab.morisia.comparch.InterfaceType#getArchitecture()
	 * @see #getInterfaceType()
	 * @generated
	 */
	EReference getInterfaceType_Architecture();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.InterfaceType#getRequiringComponentTypes <em>Requiring Component Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Requiring Component Types</em>'.
	 * @see de.mdelab.morisia.comparch.InterfaceType#getRequiringComponentTypes()
	 * @see #getInterfaceType()
	 * @generated
	 */
	EReference getInterfaceType_RequiringComponentTypes();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.InterfaceType#getProvidingComponentTypes <em>Providing Component Types</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Providing Component Types</em>'.
	 * @see de.mdelab.morisia.comparch.InterfaceType#getProvidingComponentTypes()
	 * @see #getInterfaceType()
	 * @generated
	 */
	EReference getInterfaceType_ProvidingComponentTypes();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.InterfaceType#getMethodSpecification <em>Method Specification</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Method Specification</em>'.
	 * @see de.mdelab.morisia.comparch.InterfaceType#getMethodSpecification()
	 * @see #getInterfaceType()
	 * @generated
	 */
	EReference getInterfaceType_MethodSpecification();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.InterfaceType#getInstances <em>Instances</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Instances</em>'.
	 * @see de.mdelab.morisia.comparch.InterfaceType#getInstances()
	 * @see #getInterfaceType()
	 * @generated
	 */
	EReference getInterfaceType_Instances();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.ParameterType <em>Parameter Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Parameter Type</em>'.
	 * @see de.mdelab.morisia.comparch.ParameterType
	 * @generated
	 */
	EClass getParameterType();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ParameterType#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see de.mdelab.morisia.comparch.ParameterType#getName()
	 * @see #getParameterType()
	 * @generated
	 */
	EAttribute getParameterType_Name();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ParameterType#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Type</em>'.
	 * @see de.mdelab.morisia.comparch.ParameterType#getType()
	 * @see #getParameterType()
	 * @generated
	 */
	EAttribute getParameterType_Type();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.ParameterType#getDefaultValue <em>Default Value</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Default Value</em>'.
	 * @see de.mdelab.morisia.comparch.ParameterType#getDefaultValue()
	 * @see #getParameterType()
	 * @generated
	 */
	EAttribute getParameterType_DefaultValue();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.ParameterType#getComponentType <em>Component Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Component Type</em>'.
	 * @see de.mdelab.morisia.comparch.ParameterType#getComponentType()
	 * @see #getParameterType()
	 * @generated
	 */
	EReference getParameterType_ComponentType();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.ParameterType#getInstances <em>Instances</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Instances</em>'.
	 * @see de.mdelab.morisia.comparch.ParameterType#getInstances()
	 * @see #getParameterType()
	 * @generated
	 */
	EReference getParameterType_Instances();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Component <em>Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Component</em>'.
	 * @see de.mdelab.morisia.comparch.Component
	 * @generated
	 */
	EClass getComponent();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.Component#getParameters <em>Parameters</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Parameters</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getParameters()
	 * @see #getComponent()
	 * @generated
	 */
	EReference getComponent_Parameters();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.Component#getRequiredInterfaces <em>Required Interfaces</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Required Interfaces</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getRequiredInterfaces()
	 * @see #getComponent()
	 * @generated
	 */
	EReference getComponent_RequiredInterfaces();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Component#getState <em>State</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>State</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getState()
	 * @see #getComponent()
	 * @generated
	 */
	EAttribute getComponent_State();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.Component#getProvidedInterfaces <em>Provided Interfaces</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Provided Interfaces</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getProvidedInterfaces()
	 * @see #getComponent()
	 * @generated
	 */
	EReference getComponent_ProvidedInterfaces();

	/**
	 * Returns the meta object for the reference '{@link de.mdelab.morisia.comparch.Component#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Type</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getType()
	 * @see #getComponent()
	 * @generated
	 */
	EReference getComponent_Type();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.Component#getMonitoredProperties <em>Monitored Properties</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Monitored Properties</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getMonitoredProperties()
	 * @see #getComponent()
	 * @generated
	 */
	EReference getComponent_MonitoredProperties();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.Component#getTenant <em>Tenant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Tenant</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getTenant()
	 * @see #getComponent()
	 * @generated
	 */
	EReference getComponent_Tenant();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.Component#getIssues <em>Issues</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Issues</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getIssues()
	 * @see #getComponent()
	 * @generated
	 */
	EReference getComponent_Issues();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Component#getInUseReplica <em>In Use Replica</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>In Use Replica</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getInUseReplica()
	 * @see #getComponent()
	 * @generated
	 */
	EAttribute getComponent_InUseReplica();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Component#getHealthStatus <em>Health Status</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Health Status</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getHealthStatus()
	 * @see #getComponent()
	 * @generated
	 */
	EAttribute getComponent_HealthStatus();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Component#getPerformanceOld <em>Performance Old</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Performance Old</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getPerformanceOld()
	 * @see #getComponent()
	 * @generated
	 */
	EAttribute getComponent_PerformanceOld();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Component#getOriginalRequest <em>Original Request</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Original Request</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getOriginalRequest()
	 * @see #getComponent()
	 * @generated
	 */
	EAttribute getComponent_OriginalRequest();

	/**
	 * Returns the meta object for the reference '{@link de.mdelab.morisia.comparch.Component#getDeployedOn <em>Deployed On</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Deployed On</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getDeployedOn()
	 * @see #getComponent()
	 * @generated
	 */
	EReference getComponent_DeployedOn();

	/**
	 * Returns the meta object for the containment reference '{@link de.mdelab.morisia.comparch.Component#getReplica <em>Replica</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Replica</em>'.
	 * @see de.mdelab.morisia.comparch.Component#getReplica()
	 * @see #getComponent()
	 * @generated
	 */
	EReference getComponent_Replica();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Replica <em>Replica</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Replica</em>'.
	 * @see de.mdelab.morisia.comparch.Replica
	 * @generated
	 */
	EClass getReplica();

	/**
	 * Returns the meta object for the reference '{@link de.mdelab.morisia.comparch.Replica#getDeployedOn <em>Deployed On</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Deployed On</em>'.
	 * @see de.mdelab.morisia.comparch.Replica#getDeployedOn()
	 * @see #getReplica()
	 * @generated
	 */
	EReference getReplica_DeployedOn();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Replica#getCost <em>Cost</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Cost</em>'.
	 * @see de.mdelab.morisia.comparch.Replica#getCost()
	 * @see #getReplica()
	 * @generated
	 */
	EAttribute getReplica_Cost();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Replica#getInUse <em>In Use</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>In Use</em>'.
	 * @see de.mdelab.morisia.comparch.Replica#getInUse()
	 * @see #getReplica()
	 * @generated
	 */
	EAttribute getReplica_InUse();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Replica#getCPURequirement <em>CPU Requirement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>CPU Requirement</em>'.
	 * @see de.mdelab.morisia.comparch.Replica#getCPURequirement()
	 * @see #getReplica()
	 * @generated
	 */
	EAttribute getReplica_CPURequirement();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Replica#getMemoryRequirement <em>Memory Requirement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Memory Requirement</em>'.
	 * @see de.mdelab.morisia.comparch.Replica#getMemoryRequirement()
	 * @see #getReplica()
	 * @generated
	 */
	EAttribute getReplica_MemoryRequirement();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Replica#getStorageRequirement <em>Storage Requirement</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Storage Requirement</em>'.
	 * @see de.mdelab.morisia.comparch.Replica#getStorageRequirement()
	 * @see #getReplica()
	 * @generated
	 */
	EAttribute getReplica_StorageRequirement();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.Replica#getComponent <em>Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Component</em>'.
	 * @see de.mdelab.morisia.comparch.Replica#getComponent()
	 * @see #getReplica()
	 * @generated
	 */
	EReference getReplica_Component();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Parameter <em>Parameter</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Parameter</em>'.
	 * @see de.mdelab.morisia.comparch.Parameter
	 * @generated
	 */
	EClass getParameter();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Parameter#getValue <em>Value</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Value</em>'.
	 * @see de.mdelab.morisia.comparch.Parameter#getValue()
	 * @see #getParameter()
	 * @generated
	 */
	EAttribute getParameter_Value();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Parameter#isLazyAdaptation <em>Lazy Adaptation</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Lazy Adaptation</em>'.
	 * @see de.mdelab.morisia.comparch.Parameter#isLazyAdaptation()
	 * @see #getParameter()
	 * @generated
	 */
	EAttribute getParameter_LazyAdaptation();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.Parameter#getComponent <em>Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Component</em>'.
	 * @see de.mdelab.morisia.comparch.Parameter#getComponent()
	 * @see #getParameter()
	 * @generated
	 */
	EReference getParameter_Component();

	/**
	 * Returns the meta object for the reference '{@link de.mdelab.morisia.comparch.Parameter#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Type</em>'.
	 * @see de.mdelab.morisia.comparch.Parameter#getType()
	 * @see #getParameter()
	 * @generated
	 */
	EReference getParameter_Type();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.MethodSpecification <em>Method Specification</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Method Specification</em>'.
	 * @see de.mdelab.morisia.comparch.MethodSpecification
	 * @generated
	 */
	EClass getMethodSpecification();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.MethodSpecification#getSignature <em>Signature</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Signature</em>'.
	 * @see de.mdelab.morisia.comparch.MethodSpecification#getSignature()
	 * @see #getMethodSpecification()
	 * @generated
	 */
	EAttribute getMethodSpecification_Signature();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.MethodSpecification#getInterfaceType <em>Interface Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Interface Type</em>'.
	 * @see de.mdelab.morisia.comparch.MethodSpecification#getInterfaceType()
	 * @see #getMethodSpecification()
	 * @generated
	 */
	EReference getMethodSpecification_InterfaceType();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Interface <em>Interface</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Interface</em>'.
	 * @see de.mdelab.morisia.comparch.Interface
	 * @generated
	 */
	EClass getInterface();

	/**
	 * Returns the meta object for the reference '{@link de.mdelab.morisia.comparch.Interface#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Type</em>'.
	 * @see de.mdelab.morisia.comparch.Interface#getType()
	 * @see #getInterface()
	 * @generated
	 */
	EReference getInterface_Type();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.RequiredInterface <em>Required Interface</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Required Interface</em>'.
	 * @see de.mdelab.morisia.comparch.RequiredInterface
	 * @generated
	 */
	EClass getRequiredInterface();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.RequiredInterface#getComponent <em>Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Component</em>'.
	 * @see de.mdelab.morisia.comparch.RequiredInterface#getComponent()
	 * @see #getRequiredInterface()
	 * @generated
	 */
	EReference getRequiredInterface_Component();

	/**
	 * Returns the meta object for the containment reference '{@link de.mdelab.morisia.comparch.RequiredInterface#getConnector <em>Connector</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference '<em>Connector</em>'.
	 * @see de.mdelab.morisia.comparch.RequiredInterface#getConnector()
	 * @see #getRequiredInterface()
	 * @generated
	 */
	EReference getRequiredInterface_Connector();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.ProvidedInterface <em>Provided Interface</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Provided Interface</em>'.
	 * @see de.mdelab.morisia.comparch.ProvidedInterface
	 * @generated
	 */
	EClass getProvidedInterface();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.ProvidedInterface#getComponent <em>Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Component</em>'.
	 * @see de.mdelab.morisia.comparch.ProvidedInterface#getComponent()
	 * @see #getProvidedInterface()
	 * @generated
	 */
	EReference getProvidedInterface_Component();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.ProvidedInterface#getConnectors <em>Connectors</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Connectors</em>'.
	 * @see de.mdelab.morisia.comparch.ProvidedInterface#getConnectors()
	 * @see #getProvidedInterface()
	 * @generated
	 */
	EReference getProvidedInterface_Connectors();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.ProvidedInterface#getFailures <em>Failures</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Failures</em>'.
	 * @see de.mdelab.morisia.comparch.ProvidedInterface#getFailures()
	 * @see #getProvidedInterface()
	 * @generated
	 */
	EReference getProvidedInterface_Failures();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.ProvidedInterface#getPerformanceStats <em>Performance Stats</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Performance Stats</em>'.
	 * @see de.mdelab.morisia.comparch.ProvidedInterface#getPerformanceStats()
	 * @see #getProvidedInterface()
	 * @generated
	 */
	EReference getProvidedInterface_PerformanceStats();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Connector <em>Connector</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Connector</em>'.
	 * @see de.mdelab.morisia.comparch.Connector
	 * @generated
	 */
	EClass getConnector();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Connector#isLazyAdaptation <em>Lazy Adaptation</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Lazy Adaptation</em>'.
	 * @see de.mdelab.morisia.comparch.Connector#isLazyAdaptation()
	 * @see #getConnector()
	 * @generated
	 */
	EAttribute getConnector_LazyAdaptation();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.Connector#getSource <em>Source</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Source</em>'.
	 * @see de.mdelab.morisia.comparch.Connector#getSource()
	 * @see #getConnector()
	 * @generated
	 */
	EReference getConnector_Source();

	/**
	 * Returns the meta object for the reference '{@link de.mdelab.morisia.comparch.Connector#getTarget <em>Target</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Target</em>'.
	 * @see de.mdelab.morisia.comparch.Connector#getTarget()
	 * @see #getConnector()
	 * @generated
	 */
	EReference getConnector_Target();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Failure <em>Failure</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Failure</em>'.
	 * @see de.mdelab.morisia.comparch.Failure
	 * @generated
	 */
	EClass getFailure();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Failure#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see de.mdelab.morisia.comparch.Failure#getName()
	 * @see #getFailure()
	 * @generated
	 */
	EAttribute getFailure_Name();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Failure#getMessage <em>Message</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Message</em>'.
	 * @see de.mdelab.morisia.comparch.Failure#getMessage()
	 * @see #getFailure()
	 * @generated
	 */
	EAttribute getFailure_Message();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.Failure#getInterface <em>Interface</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Interface</em>'.
	 * @see de.mdelab.morisia.comparch.Failure#getInterface()
	 * @see #getFailure()
	 * @generated
	 */
	EReference getFailure_Interface();

	/**
	 * Returns the meta object for the reference '{@link de.mdelab.morisia.comparch.Failure#getMethod <em>Method</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Method</em>'.
	 * @see de.mdelab.morisia.comparch.Failure#getMethod()
	 * @see #getFailure()
	 * @generated
	 */
	EReference getFailure_Method();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.PerformanceStats <em>Performance Stats</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Performance Stats</em>'.
	 * @see de.mdelab.morisia.comparch.PerformanceStats
	 * @generated
	 */
	EClass getPerformanceStats();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.PerformanceStats#getMinTime <em>Min Time</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Min Time</em>'.
	 * @see de.mdelab.morisia.comparch.PerformanceStats#getMinTime()
	 * @see #getPerformanceStats()
	 * @generated
	 */
	EAttribute getPerformanceStats_MinTime();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.PerformanceStats#getMaxTime <em>Max Time</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Max Time</em>'.
	 * @see de.mdelab.morisia.comparch.PerformanceStats#getMaxTime()
	 * @see #getPerformanceStats()
	 * @generated
	 */
	EAttribute getPerformanceStats_MaxTime();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.PerformanceStats#getTotalTime <em>Total Time</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Total Time</em>'.
	 * @see de.mdelab.morisia.comparch.PerformanceStats#getTotalTime()
	 * @see #getPerformanceStats()
	 * @generated
	 */
	EAttribute getPerformanceStats_TotalTime();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.PerformanceStats#getInvocationCount <em>Invocation Count</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Invocation Count</em>'.
	 * @see de.mdelab.morisia.comparch.PerformanceStats#getInvocationCount()
	 * @see #getPerformanceStats()
	 * @generated
	 */
	EAttribute getPerformanceStats_InvocationCount();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.PerformanceStats#getInterface <em>Interface</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Interface</em>'.
	 * @see de.mdelab.morisia.comparch.PerformanceStats#getInterface()
	 * @see #getPerformanceStats()
	 * @generated
	 */
	EReference getPerformanceStats_Interface();

	/**
	 * Returns the meta object for the reference '{@link de.mdelab.morisia.comparch.PerformanceStats#getMethod <em>Method</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Method</em>'.
	 * @see de.mdelab.morisia.comparch.PerformanceStats#getMethod()
	 * @see #getPerformanceStats()
	 * @generated
	 */
	EReference getPerformanceStats_Method();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.MonitoredProperty <em>Monitored Property</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Monitored Property</em>'.
	 * @see de.mdelab.morisia.comparch.MonitoredProperty
	 * @generated
	 */
	EClass getMonitoredProperty();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.MonitoredProperty#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see de.mdelab.morisia.comparch.MonitoredProperty#getName()
	 * @see #getMonitoredProperty()
	 * @generated
	 */
	EAttribute getMonitoredProperty_Name();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.MonitoredProperty#getValue <em>Value</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Value</em>'.
	 * @see de.mdelab.morisia.comparch.MonitoredProperty#getValue()
	 * @see #getMonitoredProperty()
	 * @generated
	 */
	EAttribute getMonitoredProperty_Value();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.MonitoredProperty#getType <em>Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Type</em>'.
	 * @see de.mdelab.morisia.comparch.MonitoredProperty#getType()
	 * @see #getMonitoredProperty()
	 * @generated
	 */
	EAttribute getMonitoredProperty_Type();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.MonitoredProperty#getUnit <em>Unit</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Unit</em>'.
	 * @see de.mdelab.morisia.comparch.MonitoredProperty#getUnit()
	 * @see #getMonitoredProperty()
	 * @generated
	 */
	EAttribute getMonitoredProperty_Unit();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.MonitoredProperty#getDescription <em>Description</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Description</em>'.
	 * @see de.mdelab.morisia.comparch.MonitoredProperty#getDescription()
	 * @see #getMonitoredProperty()
	 * @generated
	 */
	EAttribute getMonitoredProperty_Description();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.MonitoredProperty#getComponent <em>Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Component</em>'.
	 * @see de.mdelab.morisia.comparch.MonitoredProperty#getComponent()
	 * @see #getMonitoredProperty()
	 * @generated
	 */
	EReference getMonitoredProperty_Component();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.MonitoredProperty#getConcernedElements <em>Concerned Elements</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Concerned Elements</em>'.
	 * @see de.mdelab.morisia.comparch.MonitoredProperty#getConcernedElements()
	 * @see #getMonitoredProperty()
	 * @generated
	 */
	EReference getMonitoredProperty_ConcernedElements();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Tenant <em>Tenant</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Tenant</em>'.
	 * @see de.mdelab.morisia.comparch.Tenant
	 * @generated
	 */
	EClass getTenant();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Tenant#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see de.mdelab.morisia.comparch.Tenant#getName()
	 * @see #getTenant()
	 * @generated
	 */
	EAttribute getTenant_Name();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.Tenant#getArchitecture <em>Architecture</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Architecture</em>'.
	 * @see de.mdelab.morisia.comparch.Tenant#getArchitecture()
	 * @see #getTenant()
	 * @generated
	 */
	EReference getTenant_Architecture();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.Tenant#getComponents <em>Components</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Components</em>'.
	 * @see de.mdelab.morisia.comparch.Tenant#getComponents()
	 * @see #getTenant()
	 * @generated
	 */
	EReference getTenant_Components();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Tenant#getReplicaFactor <em>Replica Factor</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Replica Factor</em>'.
	 * @see de.mdelab.morisia.comparch.Tenant#getReplicaFactor()
	 * @see #getTenant()
	 * @generated
	 */
	EAttribute getTenant_ReplicaFactor();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Annotations <em>Annotations</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Annotations</em>'.
	 * @see de.mdelab.morisia.comparch.Annotations
	 * @generated
	 */
	EClass getAnnotations();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.Annotations#getArchitecture <em>Architecture</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Architecture</em>'.
	 * @see de.mdelab.morisia.comparch.Annotations#getArchitecture()
	 * @see #getAnnotations()
	 * @generated
	 */
	EReference getAnnotations_Architecture();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.Annotations#getIssues <em>Issues</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Issues</em>'.
	 * @see de.mdelab.morisia.comparch.Annotations#getIssues()
	 * @see #getAnnotations()
	 * @generated
	 */
	EReference getAnnotations_Issues();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.Annotations#getRules <em>Rules</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Rules</em>'.
	 * @see de.mdelab.morisia.comparch.Annotations#getRules()
	 * @see #getAnnotations()
	 * @generated
	 */
	EReference getAnnotations_Rules();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Issue <em>Issue</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Issue</em>'.
	 * @see de.mdelab.morisia.comparch.Issue
	 * @generated
	 */
	EClass getIssue();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Issue#getUtilityDrop <em>Utility Drop</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Utility Drop</em>'.
	 * @see de.mdelab.morisia.comparch.Issue#getUtilityDrop()
	 * @see #getIssue()
	 * @generated
	 */
	EAttribute getIssue_UtilityDrop();

	/**
	 * Returns the meta object for the reference '{@link de.mdelab.morisia.comparch.Issue#getAffectedComponent <em>Affected Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Affected Component</em>'.
	 * @see de.mdelab.morisia.comparch.Issue#getAffectedComponent()
	 * @see #getIssue()
	 * @generated
	 */
	EReference getIssue_AffectedComponent();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.Issue#getAnnotations <em>Annotations</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Annotations</em>'.
	 * @see de.mdelab.morisia.comparch.Issue#getAnnotations()
	 * @see #getIssue()
	 * @generated
	 */
	EReference getIssue_Annotations();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.Issue#getHandledBy <em>Handled By</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Handled By</em>'.
	 * @see de.mdelab.morisia.comparch.Issue#getHandledBy()
	 * @see #getIssue()
	 * @generated
	 */
	EReference getIssue_HandledBy();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.CF <em>CF</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>CF</em>'.
	 * @see de.mdelab.morisia.comparch.CF
	 * @generated
	 */
	EClass getCF();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.CF1 <em>CF1</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>CF1</em>'.
	 * @see de.mdelab.morisia.comparch.CF1
	 * @generated
	 */
	EClass getCF1();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.CF2 <em>CF2</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>CF2</em>'.
	 * @see de.mdelab.morisia.comparch.CF2
	 * @generated
	 */
	EClass getCF2();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.CF3 <em>CF3</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>CF3</em>'.
	 * @see de.mdelab.morisia.comparch.CF3
	 * @generated
	 */
	EClass getCF3();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.CF5 <em>CF5</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>CF5</em>'.
	 * @see de.mdelab.morisia.comparch.CF5
	 * @generated
	 */
	EClass getCF5();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Rule <em>Rule</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Rule</em>'.
	 * @see de.mdelab.morisia.comparch.Rule
	 * @generated
	 */
	EClass getRule();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Rule#getCosts <em>Costs</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Costs</em>'.
	 * @see de.mdelab.morisia.comparch.Rule#getCosts()
	 * @see #getRule()
	 * @generated
	 */
	EAttribute getRule_Costs();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.Rule#getAnnotations <em>Annotations</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Annotations</em>'.
	 * @see de.mdelab.morisia.comparch.Rule#getAnnotations()
	 * @see #getRule()
	 * @generated
	 */
	EReference getRule_Annotations();

	/**
	 * Returns the meta object for the reference '{@link de.mdelab.morisia.comparch.Rule#getHandles <em>Handles</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Handles</em>'.
	 * @see de.mdelab.morisia.comparch.Rule#getHandles()
	 * @see #getRule()
	 * @generated
	 */
	EReference getRule_Handles();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Rule#getUtilityIncrease <em>Utility Increase</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Utility Increase</em>'.
	 * @see de.mdelab.morisia.comparch.Rule#getUtilityIncrease()
	 * @see #getRule()
	 * @generated
	 */
	EAttribute getRule_UtilityIncrease();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.MoveReplica <em>Move Replica</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Move Replica</em>'.
	 * @see de.mdelab.morisia.comparch.MoveReplica
	 * @generated
	 */
	EClass getMoveReplica();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.AddReplica <em>Add Replica</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Add Replica</em>'.
	 * @see de.mdelab.morisia.comparch.AddReplica
	 * @generated
	 */
	EClass getAddReplica();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.RemoveReplica <em>Remove Replica</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Remove Replica</em>'.
	 * @see de.mdelab.morisia.comparch.RemoveReplica
	 * @generated
	 */
	EClass getRemoveReplica();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.HwRedeployComponent <em>Hw Redeploy Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Hw Redeploy Component</em>'.
	 * @see de.mdelab.morisia.comparch.HwRedeployComponent
	 * @generated
	 */
	EClass getHwRedeployComponent();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.LwRedeployComponent <em>Lw Redeploy Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Lw Redeploy Component</em>'.
	 * @see de.mdelab.morisia.comparch.LwRedeployComponent
	 * @generated
	 */
	EClass getLwRedeployComponent();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.RestartComponent <em>Restart Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Restart Component</em>'.
	 * @see de.mdelab.morisia.comparch.RestartComponent
	 * @generated
	 */
	EClass getRestartComponent();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.ReplaceComponent <em>Replace Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Replace Component</em>'.
	 * @see de.mdelab.morisia.comparch.ReplaceComponent
	 * @generated
	 */
	EClass getReplaceComponent();

	/**
	 * Returns the meta object for the reference '{@link de.mdelab.morisia.comparch.ReplaceComponent#getAlternativeComponentType <em>Alternative Component Type</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference '<em>Alternative Component Type</em>'.
	 * @see de.mdelab.morisia.comparch.ReplaceComponent#getAlternativeComponentType()
	 * @see #getReplaceComponent()
	 * @generated
	 */
	EReference getReplaceComponent_AlternativeComponentType();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.CF4 <em>CF4</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>CF4</em>'.
	 * @see de.mdelab.morisia.comparch.CF4
	 * @generated
	 */
	EClass getCF4();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.RecreateConnector <em>Recreate Connector</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Recreate Connector</em>'.
	 * @see de.mdelab.morisia.comparch.RecreateConnector
	 * @generated
	 */
	EClass getRecreateConnector();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Server <em>Server</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Server</em>'.
	 * @see de.mdelab.morisia.comparch.Server
	 * @generated
	 */
	EClass getServer();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Server#getCPUCapacity <em>CPU Capacity</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>CPU Capacity</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getCPUCapacity()
	 * @see #getServer()
	 * @generated
	 */
	EAttribute getServer_CPUCapacity();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Server#getMemoryCapacity <em>Memory Capacity</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Memory Capacity</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getMemoryCapacity()
	 * @see #getServer()
	 * @generated
	 */
	EAttribute getServer_MemoryCapacity();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Server#getStorageCapacity <em>Storage Capacity</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Storage Capacity</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getStorageCapacity()
	 * @see #getServer()
	 * @generated
	 */
	EAttribute getServer_StorageCapacity();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Server#getCPUConsumption <em>CPU Consumption</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>CPU Consumption</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getCPUConsumption()
	 * @see #getServer()
	 * @generated
	 */
	EAttribute getServer_CPUConsumption();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Server#getMemoryConsumption <em>Memory Consumption</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Memory Consumption</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getMemoryConsumption()
	 * @see #getServer()
	 * @generated
	 */
	EAttribute getServer_MemoryConsumption();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Server#getStorageConsumption <em>Storage Consumption</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Storage Consumption</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getStorageConsumption()
	 * @see #getServer()
	 * @generated
	 */
	EAttribute getServer_StorageConsumption();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Server#getCost <em>Cost</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Cost</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getCost()
	 * @see #getServer()
	 * @generated
	 */
	EAttribute getServer_Cost();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Server#getCpuFactor <em>Cpu Factor</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Cpu Factor</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getCpuFactor()
	 * @see #getServer()
	 * @generated
	 */
	EAttribute getServer_CpuFactor();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Server#getMemoryFactor <em>Memory Factor</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Memory Factor</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getMemoryFactor()
	 * @see #getServer()
	 * @generated
	 */
	EAttribute getServer_MemoryFactor();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Server#getStorageFactor <em>Storage Factor</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Storage Factor</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getStorageFactor()
	 * @see #getServer()
	 * @generated
	 */
	EAttribute getServer_StorageFactor();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.Server#getCloud <em>Cloud</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Cloud</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getCloud()
	 * @see #getServer()
	 * @generated
	 */
	EReference getServer_Cloud();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Server#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getName()
	 * @see #getServer()
	 * @generated
	 */
	EAttribute getServer_Name();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.Server#getDeployedReplicas <em>Deployed Replicas</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Deployed Replicas</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getDeployedReplicas()
	 * @see #getServer()
	 * @generated
	 */
	EReference getServer_DeployedReplicas();

	/**
	 * Returns the meta object for the reference list '{@link de.mdelab.morisia.comparch.Server#getDeployedComponent <em>Deployed Component</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the reference list '<em>Deployed Component</em>'.
	 * @see de.mdelab.morisia.comparch.Server#getDeployedComponent()
	 * @see #getServer()
	 * @generated
	 */
	EReference getServer_DeployedComponent();

	/**
	 * Returns the meta object for class '{@link de.mdelab.morisia.comparch.Cloud <em>Cloud</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for class '<em>Cloud</em>'.
	 * @see de.mdelab.morisia.comparch.Cloud
	 * @generated
	 */
	EClass getCloud();

	/**
	 * Returns the meta object for the containment reference list '{@link de.mdelab.morisia.comparch.Cloud#getServers <em>Servers</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the containment reference list '<em>Servers</em>'.
	 * @see de.mdelab.morisia.comparch.Cloud#getServers()
	 * @see #getCloud()
	 * @generated
	 */
	EReference getCloud_Servers();

	/**
	 * Returns the meta object for the container reference '{@link de.mdelab.morisia.comparch.Cloud#getArchitecture <em>Architecture</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the container reference '<em>Architecture</em>'.
	 * @see de.mdelab.morisia.comparch.Cloud#getArchitecture()
	 * @see #getCloud()
	 * @generated
	 */
	EReference getCloud_Architecture();

	/**
	 * Returns the meta object for the attribute '{@link de.mdelab.morisia.comparch.Cloud#getName <em>Name</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for the attribute '<em>Name</em>'.
	 * @see de.mdelab.morisia.comparch.Cloud#getName()
	 * @see #getCloud()
	 * @generated
	 */
	EAttribute getCloud_Name();

	/**
	 * Returns the meta object for enum '{@link de.mdelab.morisia.comparch.ComponentLifeCycle <em>Component Life Cycle</em>}'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the meta object for enum '<em>Component Life Cycle</em>'.
	 * @see de.mdelab.morisia.comparch.ComponentLifeCycle
	 * @generated
	 */
	EEnum getComponentLifeCycle();

	/**
	 * Returns the factory that creates the instances of the model.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the factory that creates the instances of the model.
	 * @generated
	 */
	ComparchFactory getComparchFactory();

	/**
	 * <!-- begin-user-doc -->
	 * Defines literals for the meta objects that represent
	 * <ul>
	 *   <li>each class,</li>
	 *   <li>each feature of each class,</li>
	 *   <li>each enum,</li>
	 *   <li>and each data type</li>
	 * </ul>
	 * <!-- end-user-doc -->
	 * @generated
	 */
	interface Literals {
		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.ArchitectureImpl <em>Architecture</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.ArchitectureImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getArchitecture()
		 * @generated
		 */
		EClass ARCHITECTURE = eINSTANCE.getArchitecture();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute ARCHITECTURE__NAME = eINSTANCE.getArchitecture_Name();

		/**
		 * The meta object literal for the '<em><b>Component Types</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ARCHITECTURE__COMPONENT_TYPES = eINSTANCE.getArchitecture_ComponentTypes();

		/**
		 * The meta object literal for the '<em><b>Interface Types</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ARCHITECTURE__INTERFACE_TYPES = eINSTANCE.getArchitecture_InterfaceTypes();

		/**
		 * The meta object literal for the '<em><b>Tenants</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ARCHITECTURE__TENANTS = eINSTANCE.getArchitecture_Tenants();

		/**
		 * The meta object literal for the '<em><b>Cloud</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ARCHITECTURE__CLOUD = eINSTANCE.getArchitecture_Cloud();

		/**
		 * The meta object literal for the '<em><b>Annotations</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ARCHITECTURE__ANNOTATIONS = eINSTANCE.getArchitecture_Annotations();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.ArchitecturalElementImpl <em>Architectural Element</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.ArchitecturalElementImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getArchitecturalElement()
		 * @generated
		 */
		EClass ARCHITECTURAL_ELEMENT = eINSTANCE.getArchitecturalElement();

		/**
		 * The meta object literal for the '<em><b>Uid</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute ARCHITECTURAL_ELEMENT__UID = eINSTANCE.getArchitecturalElement_Uid();

		/**
		 * The meta object literal for the '<em><b>Criticality</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute ARCHITECTURAL_ELEMENT__CRITICALITY = eINSTANCE.getArchitecturalElement_Criticality();

		/**
		 * The meta object literal for the '<em><b>Importance</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute ARCHITECTURAL_ELEMENT__IMPORTANCE = eINSTANCE.getArchitecturalElement_Importance();

		/**
		 * The meta object literal for the '<em><b>ADT</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute ARCHITECTURAL_ELEMENT__ADT = eINSTANCE.getArchitecturalElement_ADT();

		/**
		 * The meta object literal for the '<em><b>Efficiency</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute ARCHITECTURAL_ELEMENT__EFFICIENCY = eINSTANCE.getArchitecturalElement_Efficiency();

		/**
		 * The meta object literal for the '<em><b>Performance</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute ARCHITECTURAL_ELEMENT__PERFORMANCE = eINSTANCE.getArchitecturalElement_Performance();

		/**
		 * The meta object literal for the '<em><b>Request</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute ARCHITECTURAL_ELEMENT__REQUEST = eINSTANCE.getArchitecturalElement_Request();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl <em>Component Type</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.ComponentTypeImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getComponentType()
		 * @generated
		 */
		EClass COMPONENT_TYPE = eINSTANCE.getComponentType();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT_TYPE__NAME = eINSTANCE.getComponentType_Name();

		/**
		 * The meta object literal for the '<em><b>Architecture</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT_TYPE__ARCHITECTURE = eINSTANCE.getComponentType_Architecture();

		/**
		 * The meta object literal for the '<em><b>Required Interface Types</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT_TYPE__REQUIRED_INTERFACE_TYPES = eINSTANCE.getComponentType_RequiredInterfaceTypes();

		/**
		 * The meta object literal for the '<em><b>Provided Interface Types</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT_TYPE__PROVIDED_INTERFACE_TYPES = eINSTANCE.getComponentType_ProvidedInterfaceTypes();

		/**
		 * The meta object literal for the '<em><b>Parameter Types</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT_TYPE__PARAMETER_TYPES = eINSTANCE.getComponentType_ParameterTypes();

		/**
		 * The meta object literal for the '<em><b>Instances</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT_TYPE__INSTANCES = eINSTANCE.getComponentType_Instances();

		/**
		 * The meta object literal for the '<em><b>Reliability</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT_TYPE__RELIABILITY = eINSTANCE.getComponentType_Reliability();

		/**
		 * The meta object literal for the '<em><b>Sat Point</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT_TYPE__SAT_POINT = eINSTANCE.getComponentType_SatPoint();

		/**
		 * The meta object literal for the '<em><b>Performance Max</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT_TYPE__PERFORMANCE_MAX = eINSTANCE.getComponentType_PerformanceMax();

		/**
		 * The meta object literal for the '<em><b>Replica Cost</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT_TYPE__REPLICA_COST = eINSTANCE.getComponentType_ReplicaCost();

		/**
		 * The meta object literal for the '<em><b>Replica ADT</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT_TYPE__REPLICA_ADT = eINSTANCE.getComponentType_ReplicaADT();

		/**
		 * The meta object literal for the '<em><b>Cpu Factor</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT_TYPE__CPU_FACTOR = eINSTANCE.getComponentType_CpuFactor();

		/**
		 * The meta object literal for the '<em><b>Memory Factor</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT_TYPE__MEMORY_FACTOR = eINSTANCE.getComponentType_MemoryFactor();

		/**
		 * The meta object literal for the '<em><b>Storage Factor</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT_TYPE__STORAGE_FACTOR = eINSTANCE.getComponentType_StorageFactor();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.InterfaceTypeImpl <em>Interface Type</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.InterfaceTypeImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getInterfaceType()
		 * @generated
		 */
		EClass INTERFACE_TYPE = eINSTANCE.getInterfaceType();

		/**
		 * The meta object literal for the '<em><b>Fully Qualified Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute INTERFACE_TYPE__FULLY_QUALIFIED_NAME = eINSTANCE.getInterfaceType_FullyQualifiedName();

		/**
		 * The meta object literal for the '<em><b>Architecture</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference INTERFACE_TYPE__ARCHITECTURE = eINSTANCE.getInterfaceType_Architecture();

		/**
		 * The meta object literal for the '<em><b>Requiring Component Types</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference INTERFACE_TYPE__REQUIRING_COMPONENT_TYPES = eINSTANCE.getInterfaceType_RequiringComponentTypes();

		/**
		 * The meta object literal for the '<em><b>Providing Component Types</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference INTERFACE_TYPE__PROVIDING_COMPONENT_TYPES = eINSTANCE.getInterfaceType_ProvidingComponentTypes();

		/**
		 * The meta object literal for the '<em><b>Method Specification</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference INTERFACE_TYPE__METHOD_SPECIFICATION = eINSTANCE.getInterfaceType_MethodSpecification();

		/**
		 * The meta object literal for the '<em><b>Instances</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference INTERFACE_TYPE__INSTANCES = eINSTANCE.getInterfaceType_Instances();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.ParameterTypeImpl <em>Parameter Type</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.ParameterTypeImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getParameterType()
		 * @generated
		 */
		EClass PARAMETER_TYPE = eINSTANCE.getParameterType();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PARAMETER_TYPE__NAME = eINSTANCE.getParameterType_Name();

		/**
		 * The meta object literal for the '<em><b>Type</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PARAMETER_TYPE__TYPE = eINSTANCE.getParameterType_Type();

		/**
		 * The meta object literal for the '<em><b>Default Value</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PARAMETER_TYPE__DEFAULT_VALUE = eINSTANCE.getParameterType_DefaultValue();

		/**
		 * The meta object literal for the '<em><b>Component Type</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PARAMETER_TYPE__COMPONENT_TYPE = eINSTANCE.getParameterType_ComponentType();

		/**
		 * The meta object literal for the '<em><b>Instances</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PARAMETER_TYPE__INSTANCES = eINSTANCE.getParameterType_Instances();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.ComponentImpl <em>Component</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.ComponentImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getComponent()
		 * @generated
		 */
		EClass COMPONENT = eINSTANCE.getComponent();

		/**
		 * The meta object literal for the '<em><b>Parameters</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT__PARAMETERS = eINSTANCE.getComponent_Parameters();

		/**
		 * The meta object literal for the '<em><b>Required Interfaces</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT__REQUIRED_INTERFACES = eINSTANCE.getComponent_RequiredInterfaces();

		/**
		 * The meta object literal for the '<em><b>State</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT__STATE = eINSTANCE.getComponent_State();

		/**
		 * The meta object literal for the '<em><b>Provided Interfaces</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT__PROVIDED_INTERFACES = eINSTANCE.getComponent_ProvidedInterfaces();

		/**
		 * The meta object literal for the '<em><b>Type</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT__TYPE = eINSTANCE.getComponent_Type();

		/**
		 * The meta object literal for the '<em><b>Monitored Properties</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT__MONITORED_PROPERTIES = eINSTANCE.getComponent_MonitoredProperties();

		/**
		 * The meta object literal for the '<em><b>Tenant</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT__TENANT = eINSTANCE.getComponent_Tenant();

		/**
		 * The meta object literal for the '<em><b>Issues</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT__ISSUES = eINSTANCE.getComponent_Issues();

		/**
		 * The meta object literal for the '<em><b>In Use Replica</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT__IN_USE_REPLICA = eINSTANCE.getComponent_InUseReplica();

		/**
		 * The meta object literal for the '<em><b>Health Status</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT__HEALTH_STATUS = eINSTANCE.getComponent_HealthStatus();

		/**
		 * The meta object literal for the '<em><b>Performance Old</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT__PERFORMANCE_OLD = eINSTANCE.getComponent_PerformanceOld();

		/**
		 * The meta object literal for the '<em><b>Original Request</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute COMPONENT__ORIGINAL_REQUEST = eINSTANCE.getComponent_OriginalRequest();

		/**
		 * The meta object literal for the '<em><b>Deployed On</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT__DEPLOYED_ON = eINSTANCE.getComponent_DeployedOn();

		/**
		 * The meta object literal for the '<em><b>Replica</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference COMPONENT__REPLICA = eINSTANCE.getComponent_Replica();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.ReplicaImpl <em>Replica</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.ReplicaImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getReplica()
		 * @generated
		 */
		EClass REPLICA = eINSTANCE.getReplica();

		/**
		 * The meta object literal for the '<em><b>Deployed On</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference REPLICA__DEPLOYED_ON = eINSTANCE.getReplica_DeployedOn();

		/**
		 * The meta object literal for the '<em><b>Cost</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute REPLICA__COST = eINSTANCE.getReplica_Cost();

		/**
		 * The meta object literal for the '<em><b>In Use</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute REPLICA__IN_USE = eINSTANCE.getReplica_InUse();

		/**
		 * The meta object literal for the '<em><b>CPU Requirement</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute REPLICA__CPU_REQUIREMENT = eINSTANCE.getReplica_CPURequirement();

		/**
		 * The meta object literal for the '<em><b>Memory Requirement</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute REPLICA__MEMORY_REQUIREMENT = eINSTANCE.getReplica_MemoryRequirement();

		/**
		 * The meta object literal for the '<em><b>Storage Requirement</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute REPLICA__STORAGE_REQUIREMENT = eINSTANCE.getReplica_StorageRequirement();

		/**
		 * The meta object literal for the '<em><b>Component</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference REPLICA__COMPONENT = eINSTANCE.getReplica_Component();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.ParameterImpl <em>Parameter</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.ParameterImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getParameter()
		 * @generated
		 */
		EClass PARAMETER = eINSTANCE.getParameter();

		/**
		 * The meta object literal for the '<em><b>Value</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PARAMETER__VALUE = eINSTANCE.getParameter_Value();

		/**
		 * The meta object literal for the '<em><b>Lazy Adaptation</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PARAMETER__LAZY_ADAPTATION = eINSTANCE.getParameter_LazyAdaptation();

		/**
		 * The meta object literal for the '<em><b>Component</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PARAMETER__COMPONENT = eINSTANCE.getParameter_Component();

		/**
		 * The meta object literal for the '<em><b>Type</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PARAMETER__TYPE = eINSTANCE.getParameter_Type();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.MethodSpecificationImpl <em>Method Specification</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.MethodSpecificationImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getMethodSpecification()
		 * @generated
		 */
		EClass METHOD_SPECIFICATION = eINSTANCE.getMethodSpecification();

		/**
		 * The meta object literal for the '<em><b>Signature</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute METHOD_SPECIFICATION__SIGNATURE = eINSTANCE.getMethodSpecification_Signature();

		/**
		 * The meta object literal for the '<em><b>Interface Type</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference METHOD_SPECIFICATION__INTERFACE_TYPE = eINSTANCE.getMethodSpecification_InterfaceType();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.InterfaceImpl <em>Interface</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.InterfaceImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getInterface()
		 * @generated
		 */
		EClass INTERFACE = eINSTANCE.getInterface();

		/**
		 * The meta object literal for the '<em><b>Type</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference INTERFACE__TYPE = eINSTANCE.getInterface_Type();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.RequiredInterfaceImpl <em>Required Interface</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.RequiredInterfaceImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getRequiredInterface()
		 * @generated
		 */
		EClass REQUIRED_INTERFACE = eINSTANCE.getRequiredInterface();

		/**
		 * The meta object literal for the '<em><b>Component</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference REQUIRED_INTERFACE__COMPONENT = eINSTANCE.getRequiredInterface_Component();

		/**
		 * The meta object literal for the '<em><b>Connector</b></em>' containment reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference REQUIRED_INTERFACE__CONNECTOR = eINSTANCE.getRequiredInterface_Connector();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.ProvidedInterfaceImpl <em>Provided Interface</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.ProvidedInterfaceImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getProvidedInterface()
		 * @generated
		 */
		EClass PROVIDED_INTERFACE = eINSTANCE.getProvidedInterface();

		/**
		 * The meta object literal for the '<em><b>Component</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROVIDED_INTERFACE__COMPONENT = eINSTANCE.getProvidedInterface_Component();

		/**
		 * The meta object literal for the '<em><b>Connectors</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROVIDED_INTERFACE__CONNECTORS = eINSTANCE.getProvidedInterface_Connectors();

		/**
		 * The meta object literal for the '<em><b>Failures</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROVIDED_INTERFACE__FAILURES = eINSTANCE.getProvidedInterface_Failures();

		/**
		 * The meta object literal for the '<em><b>Performance Stats</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PROVIDED_INTERFACE__PERFORMANCE_STATS = eINSTANCE.getProvidedInterface_PerformanceStats();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.ConnectorImpl <em>Connector</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.ConnectorImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getConnector()
		 * @generated
		 */
		EClass CONNECTOR = eINSTANCE.getConnector();

		/**
		 * The meta object literal for the '<em><b>Lazy Adaptation</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CONNECTOR__LAZY_ADAPTATION = eINSTANCE.getConnector_LazyAdaptation();

		/**
		 * The meta object literal for the '<em><b>Source</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CONNECTOR__SOURCE = eINSTANCE.getConnector_Source();

		/**
		 * The meta object literal for the '<em><b>Target</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CONNECTOR__TARGET = eINSTANCE.getConnector_Target();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.FailureImpl <em>Failure</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.FailureImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getFailure()
		 * @generated
		 */
		EClass FAILURE = eINSTANCE.getFailure();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute FAILURE__NAME = eINSTANCE.getFailure_Name();

		/**
		 * The meta object literal for the '<em><b>Message</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute FAILURE__MESSAGE = eINSTANCE.getFailure_Message();

		/**
		 * The meta object literal for the '<em><b>Interface</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference FAILURE__INTERFACE = eINSTANCE.getFailure_Interface();

		/**
		 * The meta object literal for the '<em><b>Method</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference FAILURE__METHOD = eINSTANCE.getFailure_Method();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.PerformanceStatsImpl <em>Performance Stats</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.PerformanceStatsImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getPerformanceStats()
		 * @generated
		 */
		EClass PERFORMANCE_STATS = eINSTANCE.getPerformanceStats();

		/**
		 * The meta object literal for the '<em><b>Min Time</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PERFORMANCE_STATS__MIN_TIME = eINSTANCE.getPerformanceStats_MinTime();

		/**
		 * The meta object literal for the '<em><b>Max Time</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PERFORMANCE_STATS__MAX_TIME = eINSTANCE.getPerformanceStats_MaxTime();

		/**
		 * The meta object literal for the '<em><b>Total Time</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PERFORMANCE_STATS__TOTAL_TIME = eINSTANCE.getPerformanceStats_TotalTime();

		/**
		 * The meta object literal for the '<em><b>Invocation Count</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute PERFORMANCE_STATS__INVOCATION_COUNT = eINSTANCE.getPerformanceStats_InvocationCount();

		/**
		 * The meta object literal for the '<em><b>Interface</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PERFORMANCE_STATS__INTERFACE = eINSTANCE.getPerformanceStats_Interface();

		/**
		 * The meta object literal for the '<em><b>Method</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference PERFORMANCE_STATS__METHOD = eINSTANCE.getPerformanceStats_Method();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.MonitoredPropertyImpl <em>Monitored Property</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.MonitoredPropertyImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getMonitoredProperty()
		 * @generated
		 */
		EClass MONITORED_PROPERTY = eINSTANCE.getMonitoredProperty();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute MONITORED_PROPERTY__NAME = eINSTANCE.getMonitoredProperty_Name();

		/**
		 * The meta object literal for the '<em><b>Value</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute MONITORED_PROPERTY__VALUE = eINSTANCE.getMonitoredProperty_Value();

		/**
		 * The meta object literal for the '<em><b>Type</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute MONITORED_PROPERTY__TYPE = eINSTANCE.getMonitoredProperty_Type();

		/**
		 * The meta object literal for the '<em><b>Unit</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute MONITORED_PROPERTY__UNIT = eINSTANCE.getMonitoredProperty_Unit();

		/**
		 * The meta object literal for the '<em><b>Description</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute MONITORED_PROPERTY__DESCRIPTION = eINSTANCE.getMonitoredProperty_Description();

		/**
		 * The meta object literal for the '<em><b>Component</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MONITORED_PROPERTY__COMPONENT = eINSTANCE.getMonitoredProperty_Component();

		/**
		 * The meta object literal for the '<em><b>Concerned Elements</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference MONITORED_PROPERTY__CONCERNED_ELEMENTS = eINSTANCE.getMonitoredProperty_ConcernedElements();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.TenantImpl <em>Tenant</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.TenantImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getTenant()
		 * @generated
		 */
		EClass TENANT = eINSTANCE.getTenant();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute TENANT__NAME = eINSTANCE.getTenant_Name();

		/**
		 * The meta object literal for the '<em><b>Architecture</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference TENANT__ARCHITECTURE = eINSTANCE.getTenant_Architecture();

		/**
		 * The meta object literal for the '<em><b>Components</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference TENANT__COMPONENTS = eINSTANCE.getTenant_Components();

		/**
		 * The meta object literal for the '<em><b>Replica Factor</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute TENANT__REPLICA_FACTOR = eINSTANCE.getTenant_ReplicaFactor();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.AnnotationsImpl <em>Annotations</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.AnnotationsImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getAnnotations()
		 * @generated
		 */
		EClass ANNOTATIONS = eINSTANCE.getAnnotations();

		/**
		 * The meta object literal for the '<em><b>Architecture</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ANNOTATIONS__ARCHITECTURE = eINSTANCE.getAnnotations_Architecture();

		/**
		 * The meta object literal for the '<em><b>Issues</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ANNOTATIONS__ISSUES = eINSTANCE.getAnnotations_Issues();

		/**
		 * The meta object literal for the '<em><b>Rules</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ANNOTATIONS__RULES = eINSTANCE.getAnnotations_Rules();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.IssueImpl <em>Issue</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.IssueImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getIssue()
		 * @generated
		 */
		EClass ISSUE = eINSTANCE.getIssue();

		/**
		 * The meta object literal for the '<em><b>Utility Drop</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute ISSUE__UTILITY_DROP = eINSTANCE.getIssue_UtilityDrop();

		/**
		 * The meta object literal for the '<em><b>Affected Component</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ISSUE__AFFECTED_COMPONENT = eINSTANCE.getIssue_AffectedComponent();

		/**
		 * The meta object literal for the '<em><b>Annotations</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ISSUE__ANNOTATIONS = eINSTANCE.getIssue_Annotations();

		/**
		 * The meta object literal for the '<em><b>Handled By</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference ISSUE__HANDLED_BY = eINSTANCE.getIssue_HandledBy();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.CFImpl <em>CF</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.CFImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCF()
		 * @generated
		 */
		EClass CF = eINSTANCE.getCF();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.CF1Impl <em>CF1</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.CF1Impl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCF1()
		 * @generated
		 */
		EClass CF1 = eINSTANCE.getCF1();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.CF2Impl <em>CF2</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.CF2Impl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCF2()
		 * @generated
		 */
		EClass CF2 = eINSTANCE.getCF2();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.CF3Impl <em>CF3</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.CF3Impl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCF3()
		 * @generated
		 */
		EClass CF3 = eINSTANCE.getCF3();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.CF5Impl <em>CF5</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.CF5Impl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCF5()
		 * @generated
		 */
		EClass CF5 = eINSTANCE.getCF5();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.RuleImpl <em>Rule</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.RuleImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getRule()
		 * @generated
		 */
		EClass RULE = eINSTANCE.getRule();

		/**
		 * The meta object literal for the '<em><b>Costs</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute RULE__COSTS = eINSTANCE.getRule_Costs();

		/**
		 * The meta object literal for the '<em><b>Annotations</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference RULE__ANNOTATIONS = eINSTANCE.getRule_Annotations();

		/**
		 * The meta object literal for the '<em><b>Handles</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference RULE__HANDLES = eINSTANCE.getRule_Handles();

		/**
		 * The meta object literal for the '<em><b>Utility Increase</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute RULE__UTILITY_INCREASE = eINSTANCE.getRule_UtilityIncrease();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.MoveReplicaImpl <em>Move Replica</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.MoveReplicaImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getMoveReplica()
		 * @generated
		 */
		EClass MOVE_REPLICA = eINSTANCE.getMoveReplica();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.AddReplicaImpl <em>Add Replica</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.AddReplicaImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getAddReplica()
		 * @generated
		 */
		EClass ADD_REPLICA = eINSTANCE.getAddReplica();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.RemoveReplicaImpl <em>Remove Replica</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.RemoveReplicaImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getRemoveReplica()
		 * @generated
		 */
		EClass REMOVE_REPLICA = eINSTANCE.getRemoveReplica();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.HwRedeployComponentImpl <em>Hw Redeploy Component</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.HwRedeployComponentImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getHwRedeployComponent()
		 * @generated
		 */
		EClass HW_REDEPLOY_COMPONENT = eINSTANCE.getHwRedeployComponent();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.LwRedeployComponentImpl <em>Lw Redeploy Component</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.LwRedeployComponentImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getLwRedeployComponent()
		 * @generated
		 */
		EClass LW_REDEPLOY_COMPONENT = eINSTANCE.getLwRedeployComponent();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.RestartComponentImpl <em>Restart Component</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.RestartComponentImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getRestartComponent()
		 * @generated
		 */
		EClass RESTART_COMPONENT = eINSTANCE.getRestartComponent();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.ReplaceComponentImpl <em>Replace Component</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.ReplaceComponentImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getReplaceComponent()
		 * @generated
		 */
		EClass REPLACE_COMPONENT = eINSTANCE.getReplaceComponent();

		/**
		 * The meta object literal for the '<em><b>Alternative Component Type</b></em>' reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference REPLACE_COMPONENT__ALTERNATIVE_COMPONENT_TYPE = eINSTANCE.getReplaceComponent_AlternativeComponentType();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.CF4Impl <em>CF4</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.CF4Impl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCF4()
		 * @generated
		 */
		EClass CF4 = eINSTANCE.getCF4();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.RecreateConnectorImpl <em>Recreate Connector</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.RecreateConnectorImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getRecreateConnector()
		 * @generated
		 */
		EClass RECREATE_CONNECTOR = eINSTANCE.getRecreateConnector();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.ServerImpl <em>Server</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.ServerImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getServer()
		 * @generated
		 */
		EClass SERVER = eINSTANCE.getServer();

		/**
		 * The meta object literal for the '<em><b>CPU Capacity</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SERVER__CPU_CAPACITY = eINSTANCE.getServer_CPUCapacity();

		/**
		 * The meta object literal for the '<em><b>Memory Capacity</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SERVER__MEMORY_CAPACITY = eINSTANCE.getServer_MemoryCapacity();

		/**
		 * The meta object literal for the '<em><b>Storage Capacity</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SERVER__STORAGE_CAPACITY = eINSTANCE.getServer_StorageCapacity();

		/**
		 * The meta object literal for the '<em><b>CPU Consumption</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SERVER__CPU_CONSUMPTION = eINSTANCE.getServer_CPUConsumption();

		/**
		 * The meta object literal for the '<em><b>Memory Consumption</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SERVER__MEMORY_CONSUMPTION = eINSTANCE.getServer_MemoryConsumption();

		/**
		 * The meta object literal for the '<em><b>Storage Consumption</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SERVER__STORAGE_CONSUMPTION = eINSTANCE.getServer_StorageConsumption();

		/**
		 * The meta object literal for the '<em><b>Cost</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SERVER__COST = eINSTANCE.getServer_Cost();

		/**
		 * The meta object literal for the '<em><b>Cpu Factor</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SERVER__CPU_FACTOR = eINSTANCE.getServer_CpuFactor();

		/**
		 * The meta object literal for the '<em><b>Memory Factor</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SERVER__MEMORY_FACTOR = eINSTANCE.getServer_MemoryFactor();

		/**
		 * The meta object literal for the '<em><b>Storage Factor</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SERVER__STORAGE_FACTOR = eINSTANCE.getServer_StorageFactor();

		/**
		 * The meta object literal for the '<em><b>Cloud</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SERVER__CLOUD = eINSTANCE.getServer_Cloud();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute SERVER__NAME = eINSTANCE.getServer_Name();

		/**
		 * The meta object literal for the '<em><b>Deployed Replicas</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SERVER__DEPLOYED_REPLICAS = eINSTANCE.getServer_DeployedReplicas();

		/**
		 * The meta object literal for the '<em><b>Deployed Component</b></em>' reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference SERVER__DEPLOYED_COMPONENT = eINSTANCE.getServer_DeployedComponent();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.impl.CloudImpl <em>Cloud</em>}' class.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.impl.CloudImpl
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getCloud()
		 * @generated
		 */
		EClass CLOUD = eINSTANCE.getCloud();

		/**
		 * The meta object literal for the '<em><b>Servers</b></em>' containment reference list feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CLOUD__SERVERS = eINSTANCE.getCloud_Servers();

		/**
		 * The meta object literal for the '<em><b>Architecture</b></em>' container reference feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EReference CLOUD__ARCHITECTURE = eINSTANCE.getCloud_Architecture();

		/**
		 * The meta object literal for the '<em><b>Name</b></em>' attribute feature.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @generated
		 */
		EAttribute CLOUD__NAME = eINSTANCE.getCloud_Name();

		/**
		 * The meta object literal for the '{@link de.mdelab.morisia.comparch.ComponentLifeCycle <em>Component Life Cycle</em>}' enum.
		 * <!-- begin-user-doc -->
		 * <!-- end-user-doc -->
		 * @see de.mdelab.morisia.comparch.ComponentLifeCycle
		 * @see de.mdelab.morisia.comparch.impl.ComparchPackageImpl#getComponentLifeCycle()
		 * @generated
		 */
		EEnum COMPONENT_LIFE_CYCLE = eINSTANCE.getComponentLifeCycle();

	}

} //ComparchPackage
