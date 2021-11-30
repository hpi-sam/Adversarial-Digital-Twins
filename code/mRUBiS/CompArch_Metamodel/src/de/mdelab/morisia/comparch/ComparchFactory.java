/**
 */
package de.mdelab.morisia.comparch;

import org.eclipse.emf.ecore.EFactory;

/**
 * <!-- begin-user-doc --> The <b>Factory</b> for the model. It provides a
 * create method for each non-abstract class of the model. <!-- end-user-doc -->
 * @see de.mdelab.morisia.comparch.ComparchPackage
 * @generated
 */
public interface ComparchFactory extends EFactory {
	/**
	 * The singleton instance of the factory. <!-- begin-user-doc --> <!--
	 * end-user-doc -->
	 * 
	 * @generated NOT
	 */
	ComparchFactory eINSTANCE = new de.mdelab.morisia.comparch.DefaultComparchFactoryImpl();
	// de.mdelab.morisia.comparch.impl.ComparchFactoryImpl.init();

	/**
	 * Returns a new object of class '<em>Architecture</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Architecture</em>'.
	 * @generated
	 */
	Architecture createArchitecture();

	/**
	 * Returns a new object of class '<em>Component Type</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Component Type</em>'.
	 * @generated
	 */
	ComponentType createComponentType();

	/**
	 * Returns a new object of class '<em>Interface Type</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Interface Type</em>'.
	 * @generated
	 */
	InterfaceType createInterfaceType();

	/**
	 * Returns a new object of class '<em>Parameter Type</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Parameter Type</em>'.
	 * @generated
	 */
	ParameterType createParameterType();

	/**
	 * Returns a new object of class '<em>Component</em>'.
	 * <!-- begin-user-doc
	 * --> <!-- end-user-doc -->
	 * @return a new object of class '<em>Component</em>'.
	 * @generated
	 */
	Component createComponent();

	/**
	 * Returns a new object of class '<em>Replica</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Replica</em>'.
	 * @generated
	 */
	Replica createReplica();

	/**
	 * Returns a new object of class '<em>Parameter</em>'.
	 * <!-- begin-user-doc
	 * --> <!-- end-user-doc -->
	 * @return a new object of class '<em>Parameter</em>'.
	 * @generated
	 */
	Parameter createParameter();

	/**
	 * Returns a new object of class '<em>Method Specification</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Method Specification</em>'.
	 * @generated
	 */
	MethodSpecification createMethodSpecification();

	/**
	 * Returns a new object of class '<em>Required Interface</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Required Interface</em>'.
	 * @generated
	 */
	RequiredInterface createRequiredInterface();

	/**
	 * Returns a new object of class '<em>Provided Interface</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Provided Interface</em>'.
	 * @generated
	 */
	ProvidedInterface createProvidedInterface();

	/**
	 * Returns a new object of class '<em>Connector</em>'.
	 * <!-- begin-user-doc
	 * --> <!-- end-user-doc -->
	 * @return a new object of class '<em>Connector</em>'.
	 * @generated
	 */
	Connector createConnector();

	/**
	 * Returns a new object of class '<em>Failure</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Failure</em>'.
	 * @generated
	 */
	Failure createFailure();

	/**
	 * Returns a new object of class '<em>Performance Stats</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Performance Stats</em>'.
	 * @generated
	 */
	PerformanceStats createPerformanceStats();

	/**
	 * Returns a new object of class '<em>Monitored Property</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Monitored Property</em>'.
	 * @generated
	 */
	MonitoredProperty createMonitoredProperty();

	/**
	 * Returns a new object of class '<em>Tenant</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Tenant</em>'.
	 * @generated
	 */
	Tenant createTenant();

	/**
	 * Returns a new object of class '<em>Annotations</em>'.
	 * <!-- begin-user-doc
	 * --> <!-- end-user-doc -->
	 * @return a new object of class '<em>Annotations</em>'.
	 * @generated
	 */
	Annotations createAnnotations();

	/**
	 * Returns a new object of class '<em>CF1</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>CF1</em>'.
	 * @generated
	 */
	CF1 createCF1();

	/**
	 * Returns a new object of class '<em>CF2</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>CF2</em>'.
	 * @generated
	 */
	CF2 createCF2();

	/**
	 * Returns a new object of class '<em>CF3</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>CF3</em>'.
	 * @generated
	 */
	CF3 createCF3();

	/**
	 * Returns a new object of class '<em>CF5</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>CF5</em>'.
	 * @generated
	 */
	CF5 createCF5();

	/**
	 * Returns a new object of class '<em>Move Replica</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Move Replica</em>'.
	 * @generated
	 */
	MoveReplica createMoveReplica();

	/**
	 * Returns a new object of class '<em>Add Replica</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Add Replica</em>'.
	 * @generated
	 */
	AddReplica createAddReplica();

	/**
	 * Returns a new object of class '<em>Remove Replica</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Remove Replica</em>'.
	 * @generated
	 */
	RemoveReplica createRemoveReplica();

	/**
	 * Returns a new object of class '<em>Hw Redeploy Component</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Hw Redeploy Component</em>'.
	 * @generated
	 */
	HwRedeployComponent createHwRedeployComponent();

	/**
	 * Returns a new object of class '<em>Lw Redeploy Component</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Lw Redeploy Component</em>'.
	 * @generated
	 */
	LwRedeployComponent createLwRedeployComponent();

	/**
	 * Returns a new object of class '<em>Restart Component</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Restart Component</em>'.
	 * @generated
	 */
	RestartComponent createRestartComponent();

	/**
	 * Returns a new object of class '<em>Replace Component</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Replace Component</em>'.
	 * @generated
	 */
	ReplaceComponent createReplaceComponent();

	/**
	 * Returns a new object of class '<em>CF4</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>CF4</em>'.
	 * @generated
	 */
	CF4 createCF4();

	/**
	 * Returns a new object of class '<em>Recreate Connector</em>'. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @return a new object of class '<em>Recreate Connector</em>'.
	 * @generated
	 */
	RecreateConnector createRecreateConnector();

	/**
	 * Returns a new object of class '<em>Server</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Server</em>'.
	 * @generated
	 */
	Server createServer();

	/**
	 * Returns a new object of class '<em>Cloud</em>'.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return a new object of class '<em>Cloud</em>'.
	 * @generated
	 */
	Cloud createCloud();

	/**
	 * Returns the package supported by this factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the package supported by this factory.
	 * @generated
	 */
	ComparchPackage getComparchPackage();

} // ComparchFactory
