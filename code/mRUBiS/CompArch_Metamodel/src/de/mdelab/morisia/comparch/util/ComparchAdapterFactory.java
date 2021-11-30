/**
 */
package de.mdelab.morisia.comparch.util;

import de.mdelab.morisia.comparch.*;

import org.eclipse.emf.common.notify.Adapter;
import org.eclipse.emf.common.notify.Notifier;

import org.eclipse.emf.common.notify.impl.AdapterFactoryImpl;

import org.eclipse.emf.ecore.EObject;

/**
 * <!-- begin-user-doc -->
 * The <b>Adapter Factory</b> for the model.
 * It provides an adapter <code>createXXX</code> method for each class of the model.
 * <!-- end-user-doc -->
 * @see de.mdelab.morisia.comparch.ComparchPackage
 * @generated
 */
public class ComparchAdapterFactory extends AdapterFactoryImpl {
	/**
	 * The cached model package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected static ComparchPackage modelPackage;

	/**
	 * Creates an instance of the adapter factory.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComparchAdapterFactory() {
		if (modelPackage == null) {
			modelPackage = ComparchPackage.eINSTANCE;
		}
	}

	/**
	 * Returns whether this factory is applicable for the type of the object.
	 * <!-- begin-user-doc -->
	 * This implementation returns <code>true</code> if the object is either the model's package or is an instance object of the model.
	 * <!-- end-user-doc -->
	 * @return whether this factory is applicable for the type of the object.
	 * @generated
	 */
	@Override
	public boolean isFactoryForType(Object object) {
		if (object == modelPackage) {
			return true;
		}
		if (object instanceof EObject) {
			return ((EObject)object).eClass().getEPackage() == modelPackage;
		}
		return false;
	}

	/**
	 * The switch that delegates to the <code>createXXX</code> methods.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ComparchSwitch<Adapter> modelSwitch =
		new ComparchSwitch<Adapter>() {
			@Override
			public Adapter caseCloud(Cloud object) {
				return createCloudAdapter();
			}
			@Override
			public Adapter caseServer(Server object) {
				return createServerAdapter();
			}
			@Override
			public Adapter caseArchitecture(Architecture object) {
				return createArchitectureAdapter();
			}
			@Override
			public Adapter caseArchitecturalElement(ArchitecturalElement object) {
				return createArchitecturalElementAdapter();
			}
			@Override
			public Adapter caseComponentType(ComponentType object) {
				return createComponentTypeAdapter();
			}
			@Override
			public Adapter caseInterfaceType(InterfaceType object) {
				return createInterfaceTypeAdapter();
			}
			@Override
			public Adapter caseParameterType(ParameterType object) {
				return createParameterTypeAdapter();
			}
			@Override
			public Adapter caseComponent(Component object) {
				return createComponentAdapter();
			}
			@Override
			public Adapter caseReplica(Replica object) {
				return createReplicaAdapter();
			}
			@Override
			public Adapter caseParameter(Parameter object) {
				return createParameterAdapter();
			}
			@Override
			public Adapter caseMethodSpecification(MethodSpecification object) {
				return createMethodSpecificationAdapter();
			}
			@Override
			public Adapter caseInterface(Interface object) {
				return createInterfaceAdapter();
			}
			@Override
			public Adapter caseRequiredInterface(RequiredInterface object) {
				return createRequiredInterfaceAdapter();
			}
			@Override
			public Adapter caseProvidedInterface(ProvidedInterface object) {
				return createProvidedInterfaceAdapter();
			}
			@Override
			public Adapter caseConnector(Connector object) {
				return createConnectorAdapter();
			}
			@Override
			public Adapter caseFailure(Failure object) {
				return createFailureAdapter();
			}
			@Override
			public Adapter casePerformanceStats(PerformanceStats object) {
				return createPerformanceStatsAdapter();
			}
			@Override
			public Adapter caseMonitoredProperty(MonitoredProperty object) {
				return createMonitoredPropertyAdapter();
			}
			@Override
			public Adapter caseTenant(Tenant object) {
				return createTenantAdapter();
			}
			@Override
			public Adapter caseAnnotations(Annotations object) {
				return createAnnotationsAdapter();
			}
			@Override
			public Adapter caseIssue(Issue object) {
				return createIssueAdapter();
			}
			@Override
			public Adapter caseCF(CF object) {
				return createCFAdapter();
			}
			@Override
			public Adapter caseCF1(CF1 object) {
				return createCF1Adapter();
			}
			@Override
			public Adapter caseCF2(CF2 object) {
				return createCF2Adapter();
			}
			@Override
			public Adapter caseCF3(CF3 object) {
				return createCF3Adapter();
			}
			@Override
			public Adapter caseCF5(CF5 object) {
				return createCF5Adapter();
			}
			@Override
			public Adapter caseRule(Rule object) {
				return createRuleAdapter();
			}
			@Override
			public Adapter caseMoveReplica(MoveReplica object) {
				return createMoveReplicaAdapter();
			}
			@Override
			public Adapter caseAddReplica(AddReplica object) {
				return createAddReplicaAdapter();
			}
			@Override
			public Adapter caseRemoveReplica(RemoveReplica object) {
				return createRemoveReplicaAdapter();
			}
			@Override
			public Adapter caseHwRedeployComponent(HwRedeployComponent object) {
				return createHwRedeployComponentAdapter();
			}
			@Override
			public Adapter caseLwRedeployComponent(LwRedeployComponent object) {
				return createLwRedeployComponentAdapter();
			}
			@Override
			public Adapter caseRestartComponent(RestartComponent object) {
				return createRestartComponentAdapter();
			}
			@Override
			public Adapter caseReplaceComponent(ReplaceComponent object) {
				return createReplaceComponentAdapter();
			}
			@Override
			public Adapter caseCF4(CF4 object) {
				return createCF4Adapter();
			}
			@Override
			public Adapter caseRecreateConnector(RecreateConnector object) {
				return createRecreateConnectorAdapter();
			}
			@Override
			public Adapter defaultCase(EObject object) {
				return createEObjectAdapter();
			}
		};

	/**
	 * Creates an adapter for the <code>target</code>.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param target the object to adapt.
	 * @return the adapter for the <code>target</code>.
	 * @generated
	 */
	@Override
	public Adapter createAdapter(Notifier target) {
		return modelSwitch.doSwitch((EObject)target);
	}


	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Architecture <em>Architecture</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Architecture
	 * @generated
	 */
	public Adapter createArchitectureAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.ArchitecturalElement <em>Architectural Element</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.ArchitecturalElement
	 * @generated
	 */
	public Adapter createArchitecturalElementAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.ComponentType <em>Component Type</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.ComponentType
	 * @generated
	 */
	public Adapter createComponentTypeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.InterfaceType <em>Interface Type</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.InterfaceType
	 * @generated
	 */
	public Adapter createInterfaceTypeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.ParameterType <em>Parameter Type</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.ParameterType
	 * @generated
	 */
	public Adapter createParameterTypeAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Component <em>Component</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Component
	 * @generated
	 */
	public Adapter createComponentAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Replica <em>Replica</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Replica
	 * @generated
	 */
	public Adapter createReplicaAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Parameter <em>Parameter</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Parameter
	 * @generated
	 */
	public Adapter createParameterAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.MethodSpecification <em>Method Specification</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.MethodSpecification
	 * @generated
	 */
	public Adapter createMethodSpecificationAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Interface <em>Interface</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Interface
	 * @generated
	 */
	public Adapter createInterfaceAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.RequiredInterface <em>Required Interface</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.RequiredInterface
	 * @generated
	 */
	public Adapter createRequiredInterfaceAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.ProvidedInterface <em>Provided Interface</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.ProvidedInterface
	 * @generated
	 */
	public Adapter createProvidedInterfaceAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Connector <em>Connector</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Connector
	 * @generated
	 */
	public Adapter createConnectorAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Failure <em>Failure</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Failure
	 * @generated
	 */
	public Adapter createFailureAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.PerformanceStats <em>Performance Stats</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.PerformanceStats
	 * @generated
	 */
	public Adapter createPerformanceStatsAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.MonitoredProperty <em>Monitored Property</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.MonitoredProperty
	 * @generated
	 */
	public Adapter createMonitoredPropertyAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Tenant <em>Tenant</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Tenant
	 * @generated
	 */
	public Adapter createTenantAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Annotations <em>Annotations</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Annotations
	 * @generated
	 */
	public Adapter createAnnotationsAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Issue <em>Issue</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Issue
	 * @generated
	 */
	public Adapter createIssueAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.CF <em>CF</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.CF
	 * @generated
	 */
	public Adapter createCFAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.CF1 <em>CF1</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.CF1
	 * @generated
	 */
	public Adapter createCF1Adapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.CF2 <em>CF2</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.CF2
	 * @generated
	 */
	public Adapter createCF2Adapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.CF3 <em>CF3</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.CF3
	 * @generated
	 */
	public Adapter createCF3Adapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.CF5 <em>CF5</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.CF5
	 * @generated
	 */
	public Adapter createCF5Adapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Rule <em>Rule</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Rule
	 * @generated
	 */
	public Adapter createRuleAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.MoveReplica <em>Move Replica</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.MoveReplica
	 * @generated
	 */
	public Adapter createMoveReplicaAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.AddReplica <em>Add Replica</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.AddReplica
	 * @generated
	 */
	public Adapter createAddReplicaAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.RemoveReplica <em>Remove Replica</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.RemoveReplica
	 * @generated
	 */
	public Adapter createRemoveReplicaAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.HwRedeployComponent <em>Hw Redeploy Component</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.HwRedeployComponent
	 * @generated
	 */
	public Adapter createHwRedeployComponentAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.LwRedeployComponent <em>Lw Redeploy Component</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.LwRedeployComponent
	 * @generated
	 */
	public Adapter createLwRedeployComponentAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.RestartComponent <em>Restart Component</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.RestartComponent
	 * @generated
	 */
	public Adapter createRestartComponentAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.ReplaceComponent <em>Replace Component</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.ReplaceComponent
	 * @generated
	 */
	public Adapter createReplaceComponentAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.CF4 <em>CF4</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.CF4
	 * @generated
	 */
	public Adapter createCF4Adapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.RecreateConnector <em>Recreate Connector</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.RecreateConnector
	 * @generated
	 */
	public Adapter createRecreateConnectorAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Server <em>Server</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Server
	 * @generated
	 */
	public Adapter createServerAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for an object of class '{@link de.mdelab.morisia.comparch.Cloud <em>Cloud</em>}'.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null so that we can easily ignore cases;
	 * it's useful to ignore a case when inheritance will catch all the cases anyway.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @see de.mdelab.morisia.comparch.Cloud
	 * @generated
	 */
	public Adapter createCloudAdapter() {
		return null;
	}

	/**
	 * Creates a new adapter for the default case.
	 * <!-- begin-user-doc -->
	 * This default implementation returns null.
	 * <!-- end-user-doc -->
	 * @return the new adapter.
	 * @generated
	 */
	public Adapter createEObjectAdapter() {
		return null;
	}

} //ComparchAdapterFactory
