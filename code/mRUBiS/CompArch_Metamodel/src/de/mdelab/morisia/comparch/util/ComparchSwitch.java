/**
 */
package de.mdelab.morisia.comparch.util;

import de.mdelab.morisia.comparch.*;

import org.eclipse.emf.ecore.EObject;
import org.eclipse.emf.ecore.EPackage;

import org.eclipse.emf.ecore.util.Switch;

/**
 * <!-- begin-user-doc -->
 * The <b>Switch</b> for the model's inheritance hierarchy.
 * It supports the call {@link #doSwitch(EObject) doSwitch(object)}
 * to invoke the <code>caseXXX</code> method for each class of the model,
 * starting with the actual class of the object
 * and proceeding up the inheritance hierarchy
 * until a non-null result is returned,
 * which is the result of the switch.
 * <!-- end-user-doc -->
 * @see de.mdelab.morisia.comparch.ComparchPackage
 * @generated
 */
public class ComparchSwitch<T> extends Switch<T> {
	/**
	 * The cached model package
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected static ComparchPackage modelPackage;

	/**
	 * Creates an instance of the switch.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComparchSwitch() {
		if (modelPackage == null) {
			modelPackage = ComparchPackage.eINSTANCE;
		}
	}

	/**
	 * Checks whether this is a switch for the given package.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param ePackage the package in question.
	 * @return whether this is a switch for the given package.
	 * @generated
	 */
	@Override
	protected boolean isSwitchFor(EPackage ePackage) {
		return ePackage == modelPackage;
	}

	/**
	 * Calls <code>caseXXX</code> for each class of the model until one returns a non null result; it yields that result.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @return the first non-null result returned by a <code>caseXXX</code> call.
	 * @generated
	 */
	@Override
	protected T doSwitch(int classifierID, EObject theEObject) {
		switch (classifierID) {
			case ComparchPackage.CLOUD: {
				Cloud cloud = (Cloud)theEObject;
				T result = caseCloud(cloud);
				if (result == null) result = caseArchitecturalElement(cloud);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.SERVER: {
				Server server = (Server)theEObject;
				T result = caseServer(server);
				if (result == null) result = caseArchitecturalElement(server);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.ARCHITECTURE: {
				Architecture architecture = (Architecture)theEObject;
				T result = caseArchitecture(architecture);
				if (result == null) result = caseArchitecturalElement(architecture);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.ARCHITECTURAL_ELEMENT: {
				ArchitecturalElement architecturalElement = (ArchitecturalElement)theEObject;
				T result = caseArchitecturalElement(architecturalElement);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.COMPONENT_TYPE: {
				ComponentType componentType = (ComponentType)theEObject;
				T result = caseComponentType(componentType);
				if (result == null) result = caseArchitecturalElement(componentType);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.INTERFACE_TYPE: {
				InterfaceType interfaceType = (InterfaceType)theEObject;
				T result = caseInterfaceType(interfaceType);
				if (result == null) result = caseArchitecturalElement(interfaceType);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.PARAMETER_TYPE: {
				ParameterType parameterType = (ParameterType)theEObject;
				T result = caseParameterType(parameterType);
				if (result == null) result = caseArchitecturalElement(parameterType);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.COMPONENT: {
				Component component = (Component)theEObject;
				T result = caseComponent(component);
				if (result == null) result = caseArchitecturalElement(component);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.REPLICA: {
				Replica replica = (Replica)theEObject;
				T result = caseReplica(replica);
				if (result == null) result = caseArchitecturalElement(replica);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.PARAMETER: {
				Parameter parameter = (Parameter)theEObject;
				T result = caseParameter(parameter);
				if (result == null) result = caseArchitecturalElement(parameter);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.METHOD_SPECIFICATION: {
				MethodSpecification methodSpecification = (MethodSpecification)theEObject;
				T result = caseMethodSpecification(methodSpecification);
				if (result == null) result = caseArchitecturalElement(methodSpecification);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.INTERFACE: {
				Interface interface_ = (Interface)theEObject;
				T result = caseInterface(interface_);
				if (result == null) result = caseArchitecturalElement(interface_);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.REQUIRED_INTERFACE: {
				RequiredInterface requiredInterface = (RequiredInterface)theEObject;
				T result = caseRequiredInterface(requiredInterface);
				if (result == null) result = caseInterface(requiredInterface);
				if (result == null) result = caseArchitecturalElement(requiredInterface);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.PROVIDED_INTERFACE: {
				ProvidedInterface providedInterface = (ProvidedInterface)theEObject;
				T result = caseProvidedInterface(providedInterface);
				if (result == null) result = caseInterface(providedInterface);
				if (result == null) result = caseArchitecturalElement(providedInterface);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.CONNECTOR: {
				Connector connector = (Connector)theEObject;
				T result = caseConnector(connector);
				if (result == null) result = caseArchitecturalElement(connector);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.FAILURE: {
				Failure failure = (Failure)theEObject;
				T result = caseFailure(failure);
				if (result == null) result = caseArchitecturalElement(failure);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.PERFORMANCE_STATS: {
				PerformanceStats performanceStats = (PerformanceStats)theEObject;
				T result = casePerformanceStats(performanceStats);
				if (result == null) result = caseArchitecturalElement(performanceStats);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.MONITORED_PROPERTY: {
				MonitoredProperty monitoredProperty = (MonitoredProperty)theEObject;
				T result = caseMonitoredProperty(monitoredProperty);
				if (result == null) result = caseArchitecturalElement(monitoredProperty);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.TENANT: {
				Tenant tenant = (Tenant)theEObject;
				T result = caseTenant(tenant);
				if (result == null) result = caseArchitecturalElement(tenant);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.ANNOTATIONS: {
				Annotations annotations = (Annotations)theEObject;
				T result = caseAnnotations(annotations);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.ISSUE: {
				Issue issue = (Issue)theEObject;
				T result = caseIssue(issue);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.CF: {
				CF cf = (CF)theEObject;
				T result = caseCF(cf);
				if (result == null) result = caseIssue(cf);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.CF1: {
				CF1 cf1 = (CF1)theEObject;
				T result = caseCF1(cf1);
				if (result == null) result = caseCF(cf1);
				if (result == null) result = caseIssue(cf1);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.CF2: {
				CF2 cf2 = (CF2)theEObject;
				T result = caseCF2(cf2);
				if (result == null) result = caseCF(cf2);
				if (result == null) result = caseIssue(cf2);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.CF3: {
				CF3 cf3 = (CF3)theEObject;
				T result = caseCF3(cf3);
				if (result == null) result = caseCF(cf3);
				if (result == null) result = caseIssue(cf3);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.CF5: {
				CF5 cf5 = (CF5)theEObject;
				T result = caseCF5(cf5);
				if (result == null) result = caseCF(cf5);
				if (result == null) result = caseIssue(cf5);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.RULE: {
				Rule rule = (Rule)theEObject;
				T result = caseRule(rule);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.MOVE_REPLICA: {
				MoveReplica moveReplica = (MoveReplica)theEObject;
				T result = caseMoveReplica(moveReplica);
				if (result == null) result = caseRule(moveReplica);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.ADD_REPLICA: {
				AddReplica addReplica = (AddReplica)theEObject;
				T result = caseAddReplica(addReplica);
				if (result == null) result = caseRule(addReplica);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.REMOVE_REPLICA: {
				RemoveReplica removeReplica = (RemoveReplica)theEObject;
				T result = caseRemoveReplica(removeReplica);
				if (result == null) result = caseRule(removeReplica);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.HW_REDEPLOY_COMPONENT: {
				HwRedeployComponent hwRedeployComponent = (HwRedeployComponent)theEObject;
				T result = caseHwRedeployComponent(hwRedeployComponent);
				if (result == null) result = caseRule(hwRedeployComponent);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.LW_REDEPLOY_COMPONENT: {
				LwRedeployComponent lwRedeployComponent = (LwRedeployComponent)theEObject;
				T result = caseLwRedeployComponent(lwRedeployComponent);
				if (result == null) result = caseRule(lwRedeployComponent);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.RESTART_COMPONENT: {
				RestartComponent restartComponent = (RestartComponent)theEObject;
				T result = caseRestartComponent(restartComponent);
				if (result == null) result = caseRule(restartComponent);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.REPLACE_COMPONENT: {
				ReplaceComponent replaceComponent = (ReplaceComponent)theEObject;
				T result = caseReplaceComponent(replaceComponent);
				if (result == null) result = caseRule(replaceComponent);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.CF4: {
				CF4 cf4 = (CF4)theEObject;
				T result = caseCF4(cf4);
				if (result == null) result = caseCF(cf4);
				if (result == null) result = caseIssue(cf4);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			case ComparchPackage.RECREATE_CONNECTOR: {
				RecreateConnector recreateConnector = (RecreateConnector)theEObject;
				T result = caseRecreateConnector(recreateConnector);
				if (result == null) result = caseRule(recreateConnector);
				if (result == null) result = defaultCase(theEObject);
				return result;
			}
			default: return defaultCase(theEObject);
		}
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Architecture</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Architecture</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseArchitecture(Architecture object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Architectural Element</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Architectural Element</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseArchitecturalElement(ArchitecturalElement object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Component Type</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Component Type</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseComponentType(ComponentType object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Interface Type</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Interface Type</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseInterfaceType(InterfaceType object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Parameter Type</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Parameter Type</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseParameterType(ParameterType object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Component</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Component</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseComponent(Component object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Replica</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Replica</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseReplica(Replica object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Parameter</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Parameter</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseParameter(Parameter object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Method Specification</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Method Specification</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseMethodSpecification(MethodSpecification object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Interface</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Interface</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseInterface(Interface object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Required Interface</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Required Interface</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseRequiredInterface(RequiredInterface object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Provided Interface</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Provided Interface</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseProvidedInterface(ProvidedInterface object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Connector</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Connector</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseConnector(Connector object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Failure</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Failure</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseFailure(Failure object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Performance Stats</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Performance Stats</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T casePerformanceStats(PerformanceStats object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Monitored Property</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Monitored Property</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseMonitoredProperty(MonitoredProperty object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Tenant</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Tenant</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseTenant(Tenant object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Annotations</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Annotations</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseAnnotations(Annotations object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Issue</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Issue</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseIssue(Issue object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>CF</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>CF</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseCF(CF object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>CF1</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>CF1</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseCF1(CF1 object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>CF2</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>CF2</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseCF2(CF2 object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>CF3</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>CF3</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseCF3(CF3 object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>CF5</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>CF5</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseCF5(CF5 object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Rule</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Rule</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseRule(Rule object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Move Replica</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Move Replica</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseMoveReplica(MoveReplica object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Add Replica</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Add Replica</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseAddReplica(AddReplica object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Remove Replica</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Remove Replica</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseRemoveReplica(RemoveReplica object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Hw Redeploy Component</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Hw Redeploy Component</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseHwRedeployComponent(HwRedeployComponent object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Lw Redeploy Component</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Lw Redeploy Component</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseLwRedeployComponent(LwRedeployComponent object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Restart Component</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Restart Component</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseRestartComponent(RestartComponent object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Replace Component</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Replace Component</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseReplaceComponent(ReplaceComponent object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>CF4</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>CF4</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseCF4(CF4 object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Recreate Connector</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Recreate Connector</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseRecreateConnector(RecreateConnector object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Server</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Server</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseServer(Server object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>Cloud</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>Cloud</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject) doSwitch(EObject)
	 * @generated
	 */
	public T caseCloud(Cloud object) {
		return null;
	}

	/**
	 * Returns the result of interpreting the object as an instance of '<em>EObject</em>'.
	 * <!-- begin-user-doc -->
	 * This implementation returns null;
	 * returning a non-null result will terminate the switch, but this is the last case anyway.
	 * <!-- end-user-doc -->
	 * @param object the target of the switch.
	 * @return the result of interpreting the object as an instance of '<em>EObject</em>'.
	 * @see #doSwitch(org.eclipse.emf.ecore.EObject)
	 * @generated
	 */
	@Override
	public T defaultCase(EObject object) {
		return null;
	}

} //ComparchSwitch
