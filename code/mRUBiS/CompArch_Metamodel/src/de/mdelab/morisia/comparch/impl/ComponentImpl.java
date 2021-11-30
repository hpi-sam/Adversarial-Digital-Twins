/**
 */
package de.mdelab.morisia.comparch.impl;

import java.util.Collection;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.util.EObjectContainmentWithInverseEList;
import org.eclipse.emf.ecore.util.EObjectWithInverseResolvingEList;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.emf.ecore.util.InternalEList;

import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.MonitoredProperty;
import de.mdelab.morisia.comparch.Parameter;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Replica;
import de.mdelab.morisia.comparch.RequiredInterface;
import de.mdelab.morisia.comparch.Server;
import de.mdelab.morisia.comparch.Tenant;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Component</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getParameters <em>Parameters</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getRequiredInterfaces <em>Required Interfaces</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getState <em>State</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getProvidedInterfaces <em>Provided Interfaces</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getType <em>Type</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getMonitoredProperties <em>Monitored Properties</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getTenant <em>Tenant</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getIssues <em>Issues</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getInUseReplica <em>In Use Replica</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getHealthStatus <em>Health Status</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getPerformanceOld <em>Performance Old</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getOriginalRequest <em>Original Request</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getDeployedOn <em>Deployed On</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentImpl#getReplica <em>Replica</em>}</li>
 * </ul>
 *
 * @generated
 */
public class ComponentImpl extends ArchitecturalElementImpl implements Component {
	/**
	 * The cached value of the '{@link #getParameters() <em>Parameters</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getParameters()
	 * @generated
	 * @ordered
	 */
	protected EList<Parameter> parameters;

	/**
	 * The cached value of the '{@link #getRequiredInterfaces() <em>Required Interfaces</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getRequiredInterfaces()
	 * @generated
	 * @ordered
	 */
	protected EList<RequiredInterface> requiredInterfaces;

	/**
	 * The default value of the '{@link #getState() <em>State</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getState()
	 * @generated
	 * @ordered
	 */
	protected static final ComponentLifeCycle STATE_EDEFAULT = ComponentLifeCycle.DEPLOYED;

	/**
	 * The cached value of the '{@link #getState() <em>State</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getState()
	 * @generated
	 * @ordered
	 */
	protected ComponentLifeCycle state = STATE_EDEFAULT;

	/**
	 * The cached value of the '{@link #getProvidedInterfaces() <em>Provided Interfaces</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getProvidedInterfaces()
	 * @generated
	 * @ordered
	 */
	protected EList<ProvidedInterface> providedInterfaces;

	/**
	 * The cached value of the '{@link #getType() <em>Type</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getType()
	 * @generated
	 * @ordered
	 */
	protected ComponentType type;

	/**
	 * The cached value of the '{@link #getMonitoredProperties() <em>Monitored Properties</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMonitoredProperties()
	 * @generated
	 * @ordered
	 */
	protected EList<MonitoredProperty> monitoredProperties;

	/**
	 * The cached value of the '{@link #getIssues() <em>Issues</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getIssues()
	 * @generated
	 * @ordered
	 */
	protected EList<Issue> issues;

	/**
	 * The default value of the '{@link #getInUseReplica() <em>In Use Replica</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getInUseReplica()
	 * @generated
	 * @ordered
	 */
	protected static final double IN_USE_REPLICA_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getInUseReplica() <em>In Use Replica</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getInUseReplica()
	 * @generated
	 * @ordered
	 */
	protected double inUseReplica = IN_USE_REPLICA_EDEFAULT;

	/**
	 * The default value of the '{@link #getHealthStatus() <em>Health Status</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getHealthStatus()
	 * @generated
	 * @ordered
	 */
	protected static final double HEALTH_STATUS_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getHealthStatus() <em>Health Status</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getHealthStatus()
	 * @generated
	 * @ordered
	 */
	protected double healthStatus = HEALTH_STATUS_EDEFAULT;

	/**
	 * The default value of the '{@link #getPerformanceOld() <em>Performance Old</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPerformanceOld()
	 * @generated
	 * @ordered
	 */
	protected static final double PERFORMANCE_OLD_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getPerformanceOld() <em>Performance Old</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPerformanceOld()
	 * @generated
	 * @ordered
	 */
	protected double performanceOld = PERFORMANCE_OLD_EDEFAULT;

	/**
	 * The default value of the '{@link #getOriginalRequest() <em>Original Request</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOriginalRequest()
	 * @generated
	 * @ordered
	 */
	protected static final double ORIGINAL_REQUEST_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getOriginalRequest() <em>Original Request</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getOriginalRequest()
	 * @generated
	 * @ordered
	 */
	protected double originalRequest = ORIGINAL_REQUEST_EDEFAULT;

	/**
	 * The cached value of the '{@link #getDeployedOn() <em>Deployed On</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDeployedOn()
	 * @generated
	 * @ordered
	 */
	protected Server deployedOn;

	/**
	 * The cached value of the '{@link #getReplica() <em>Replica</em>}' containment reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getReplica()
	 * @generated
	 * @ordered
	 */
	protected Replica replica;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ComponentImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ComparchPackage.Literals.COMPONENT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Parameter> getParameters() {
		if (parameters == null) {
			parameters = new EObjectContainmentWithInverseEList<Parameter>(Parameter.class, this, ComparchPackage.COMPONENT__PARAMETERS, ComparchPackage.PARAMETER__COMPONENT);
		}
		return parameters;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<RequiredInterface> getRequiredInterfaces() {
		if (requiredInterfaces == null) {
			requiredInterfaces = new EObjectContainmentWithInverseEList<RequiredInterface>(RequiredInterface.class, this, ComparchPackage.COMPONENT__REQUIRED_INTERFACES, ComparchPackage.REQUIRED_INTERFACE__COMPONENT);
		}
		return requiredInterfaces;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComponentLifeCycle getState() {
		return state;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setState(ComponentLifeCycle newState) {
		ComponentLifeCycle oldState = state;
		state = newState == null ? STATE_EDEFAULT : newState;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT__STATE, oldState, state));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<ProvidedInterface> getProvidedInterfaces() {
		if (providedInterfaces == null) {
			providedInterfaces = new EObjectContainmentWithInverseEList<ProvidedInterface>(ProvidedInterface.class, this, ComparchPackage.COMPONENT__PROVIDED_INTERFACES, ComparchPackage.PROVIDED_INTERFACE__COMPONENT);
		}
		return providedInterfaces;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComponentType getType() {
		if (type != null && type.eIsProxy()) {
			InternalEObject oldType = (InternalEObject)type;
			type = (ComponentType)eResolveProxy(oldType);
			if (type != oldType) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, ComparchPackage.COMPONENT__TYPE, oldType, type));
			}
		}
		return type;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComponentType basicGetType() {
		return type;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetType(ComponentType newType, NotificationChain msgs) {
		ComponentType oldType = type;
		type = newType;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT__TYPE, oldType, newType);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setType(ComponentType newType) {
		if (newType != type) {
			NotificationChain msgs = null;
			if (type != null)
				msgs = ((InternalEObject)type).eInverseRemove(this, ComparchPackage.COMPONENT_TYPE__INSTANCES, ComponentType.class, msgs);
			if (newType != null)
				msgs = ((InternalEObject)newType).eInverseAdd(this, ComparchPackage.COMPONENT_TYPE__INSTANCES, ComponentType.class, msgs);
			msgs = basicSetType(newType, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT__TYPE, newType, newType));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<MonitoredProperty> getMonitoredProperties() {
		if (monitoredProperties == null) {
			monitoredProperties = new EObjectContainmentWithInverseEList<MonitoredProperty>(MonitoredProperty.class, this, ComparchPackage.COMPONENT__MONITORED_PROPERTIES, ComparchPackage.MONITORED_PROPERTY__COMPONENT);
		}
		return monitoredProperties;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Tenant getTenant() {
		if (eContainerFeatureID() != ComparchPackage.COMPONENT__TENANT) return null;
		return (Tenant)eInternalContainer();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetTenant(Tenant newTenant, NotificationChain msgs) {
		msgs = eBasicSetContainer((InternalEObject)newTenant, ComparchPackage.COMPONENT__TENANT, msgs);
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setTenant(Tenant newTenant) {
		if (newTenant != eInternalContainer() || (eContainerFeatureID() != ComparchPackage.COMPONENT__TENANT && newTenant != null)) {
			if (EcoreUtil.isAncestor(this, newTenant))
				throw new IllegalArgumentException("Recursive containment not allowed for " + toString());
			NotificationChain msgs = null;
			if (eInternalContainer() != null)
				msgs = eBasicRemoveFromContainer(msgs);
			if (newTenant != null)
				msgs = ((InternalEObject)newTenant).eInverseAdd(this, ComparchPackage.TENANT__COMPONENTS, Tenant.class, msgs);
			msgs = basicSetTenant(newTenant, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT__TENANT, newTenant, newTenant));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Issue> getIssues() {
		if (issues == null) {
			issues = new EObjectWithInverseResolvingEList<Issue>(Issue.class, this, ComparchPackage.COMPONENT__ISSUES, ComparchPackage.ISSUE__AFFECTED_COMPONENT);
		}
		return issues;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getInUseReplica() {
		return inUseReplica;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setInUseReplica(double newInUseReplica) {
		double oldInUseReplica = inUseReplica;
		inUseReplica = newInUseReplica;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT__IN_USE_REPLICA, oldInUseReplica, inUseReplica));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getHealthStatus() {
		return healthStatus;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setHealthStatus(double newHealthStatus) {
		double oldHealthStatus = healthStatus;
		healthStatus = newHealthStatus;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT__HEALTH_STATUS, oldHealthStatus, healthStatus));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getPerformanceOld() {
		return performanceOld;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPerformanceOld(double newPerformanceOld) {
		double oldPerformanceOld = performanceOld;
		performanceOld = newPerformanceOld;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT__PERFORMANCE_OLD, oldPerformanceOld, performanceOld));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getOriginalRequest() {
		return originalRequest;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setOriginalRequest(double newOriginalRequest) {
		double oldOriginalRequest = originalRequest;
		originalRequest = newOriginalRequest;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT__ORIGINAL_REQUEST, oldOriginalRequest, originalRequest));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Server getDeployedOn() {
		if (deployedOn != null && deployedOn.eIsProxy()) {
			InternalEObject oldDeployedOn = (InternalEObject)deployedOn;
			deployedOn = (Server)eResolveProxy(oldDeployedOn);
			if (deployedOn != oldDeployedOn) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, ComparchPackage.COMPONENT__DEPLOYED_ON, oldDeployedOn, deployedOn));
			}
		}
		return deployedOn;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Server basicGetDeployedOn() {
		return deployedOn;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetDeployedOn(Server newDeployedOn, NotificationChain msgs) {
		Server oldDeployedOn = deployedOn;
		deployedOn = newDeployedOn;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT__DEPLOYED_ON, oldDeployedOn, newDeployedOn);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setDeployedOn(Server newDeployedOn) {
		if (newDeployedOn != deployedOn) {
			NotificationChain msgs = null;
			if (deployedOn != null)
				msgs = ((InternalEObject)deployedOn).eInverseRemove(this, ComparchPackage.SERVER__DEPLOYED_COMPONENT, Server.class, msgs);
			if (newDeployedOn != null)
				msgs = ((InternalEObject)newDeployedOn).eInverseAdd(this, ComparchPackage.SERVER__DEPLOYED_COMPONENT, Server.class, msgs);
			msgs = basicSetDeployedOn(newDeployedOn, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT__DEPLOYED_ON, newDeployedOn, newDeployedOn));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Replica getReplica() {
		return replica;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetReplica(Replica newReplica, NotificationChain msgs) {
		Replica oldReplica = replica;
		replica = newReplica;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT__REPLICA, oldReplica, newReplica);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setReplica(Replica newReplica) {
		if (newReplica != replica) {
			NotificationChain msgs = null;
			if (replica != null)
				msgs = ((InternalEObject)replica).eInverseRemove(this, ComparchPackage.REPLICA__COMPONENT, Replica.class, msgs);
			if (newReplica != null)
				msgs = ((InternalEObject)newReplica).eInverseAdd(this, ComparchPackage.REPLICA__COMPONENT, Replica.class, msgs);
			msgs = basicSetReplica(newReplica, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT__REPLICA, newReplica, newReplica));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public NotificationChain eInverseAdd(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case ComparchPackage.COMPONENT__PARAMETERS:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getParameters()).basicAdd(otherEnd, msgs);
			case ComparchPackage.COMPONENT__REQUIRED_INTERFACES:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getRequiredInterfaces()).basicAdd(otherEnd, msgs);
			case ComparchPackage.COMPONENT__PROVIDED_INTERFACES:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getProvidedInterfaces()).basicAdd(otherEnd, msgs);
			case ComparchPackage.COMPONENT__TYPE:
				if (type != null)
					msgs = ((InternalEObject)type).eInverseRemove(this, ComparchPackage.COMPONENT_TYPE__INSTANCES, ComponentType.class, msgs);
				return basicSetType((ComponentType)otherEnd, msgs);
			case ComparchPackage.COMPONENT__MONITORED_PROPERTIES:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getMonitoredProperties()).basicAdd(otherEnd, msgs);
			case ComparchPackage.COMPONENT__TENANT:
				if (eInternalContainer() != null)
					msgs = eBasicRemoveFromContainer(msgs);
				return basicSetTenant((Tenant)otherEnd, msgs);
			case ComparchPackage.COMPONENT__ISSUES:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getIssues()).basicAdd(otherEnd, msgs);
			case ComparchPackage.COMPONENT__DEPLOYED_ON:
				if (deployedOn != null)
					msgs = ((InternalEObject)deployedOn).eInverseRemove(this, ComparchPackage.SERVER__DEPLOYED_COMPONENT, Server.class, msgs);
				return basicSetDeployedOn((Server)otherEnd, msgs);
			case ComparchPackage.COMPONENT__REPLICA:
				if (replica != null)
					msgs = ((InternalEObject)replica).eInverseRemove(this, EOPPOSITE_FEATURE_BASE - ComparchPackage.COMPONENT__REPLICA, null, msgs);
				return basicSetReplica((Replica)otherEnd, msgs);
		}
		return super.eInverseAdd(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case ComparchPackage.COMPONENT__PARAMETERS:
				return ((InternalEList<?>)getParameters()).basicRemove(otherEnd, msgs);
			case ComparchPackage.COMPONENT__REQUIRED_INTERFACES:
				return ((InternalEList<?>)getRequiredInterfaces()).basicRemove(otherEnd, msgs);
			case ComparchPackage.COMPONENT__PROVIDED_INTERFACES:
				return ((InternalEList<?>)getProvidedInterfaces()).basicRemove(otherEnd, msgs);
			case ComparchPackage.COMPONENT__TYPE:
				return basicSetType(null, msgs);
			case ComparchPackage.COMPONENT__MONITORED_PROPERTIES:
				return ((InternalEList<?>)getMonitoredProperties()).basicRemove(otherEnd, msgs);
			case ComparchPackage.COMPONENT__TENANT:
				return basicSetTenant(null, msgs);
			case ComparchPackage.COMPONENT__ISSUES:
				return ((InternalEList<?>)getIssues()).basicRemove(otherEnd, msgs);
			case ComparchPackage.COMPONENT__DEPLOYED_ON:
				return basicSetDeployedOn(null, msgs);
			case ComparchPackage.COMPONENT__REPLICA:
				return basicSetReplica(null, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eBasicRemoveFromContainerFeature(NotificationChain msgs) {
		switch (eContainerFeatureID()) {
			case ComparchPackage.COMPONENT__TENANT:
				return eInternalContainer().eInverseRemove(this, ComparchPackage.TENANT__COMPONENTS, Tenant.class, msgs);
		}
		return super.eBasicRemoveFromContainerFeature(msgs);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case ComparchPackage.COMPONENT__PARAMETERS:
				return getParameters();
			case ComparchPackage.COMPONENT__REQUIRED_INTERFACES:
				return getRequiredInterfaces();
			case ComparchPackage.COMPONENT__STATE:
				return getState();
			case ComparchPackage.COMPONENT__PROVIDED_INTERFACES:
				return getProvidedInterfaces();
			case ComparchPackage.COMPONENT__TYPE:
				if (resolve) return getType();
				return basicGetType();
			case ComparchPackage.COMPONENT__MONITORED_PROPERTIES:
				return getMonitoredProperties();
			case ComparchPackage.COMPONENT__TENANT:
				return getTenant();
			case ComparchPackage.COMPONENT__ISSUES:
				return getIssues();
			case ComparchPackage.COMPONENT__IN_USE_REPLICA:
				return getInUseReplica();
			case ComparchPackage.COMPONENT__HEALTH_STATUS:
				return getHealthStatus();
			case ComparchPackage.COMPONENT__PERFORMANCE_OLD:
				return getPerformanceOld();
			case ComparchPackage.COMPONENT__ORIGINAL_REQUEST:
				return getOriginalRequest();
			case ComparchPackage.COMPONENT__DEPLOYED_ON:
				if (resolve) return getDeployedOn();
				return basicGetDeployedOn();
			case ComparchPackage.COMPONENT__REPLICA:
				return getReplica();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case ComparchPackage.COMPONENT__PARAMETERS:
				getParameters().clear();
				getParameters().addAll((Collection<? extends Parameter>)newValue);
				return;
			case ComparchPackage.COMPONENT__REQUIRED_INTERFACES:
				getRequiredInterfaces().clear();
				getRequiredInterfaces().addAll((Collection<? extends RequiredInterface>)newValue);
				return;
			case ComparchPackage.COMPONENT__STATE:
				setState((ComponentLifeCycle)newValue);
				return;
			case ComparchPackage.COMPONENT__PROVIDED_INTERFACES:
				getProvidedInterfaces().clear();
				getProvidedInterfaces().addAll((Collection<? extends ProvidedInterface>)newValue);
				return;
			case ComparchPackage.COMPONENT__TYPE:
				setType((ComponentType)newValue);
				return;
			case ComparchPackage.COMPONENT__MONITORED_PROPERTIES:
				getMonitoredProperties().clear();
				getMonitoredProperties().addAll((Collection<? extends MonitoredProperty>)newValue);
				return;
			case ComparchPackage.COMPONENT__TENANT:
				setTenant((Tenant)newValue);
				return;
			case ComparchPackage.COMPONENT__ISSUES:
				getIssues().clear();
				getIssues().addAll((Collection<? extends Issue>)newValue);
				return;
			case ComparchPackage.COMPONENT__IN_USE_REPLICA:
				setInUseReplica((Double)newValue);
				return;
			case ComparchPackage.COMPONENT__HEALTH_STATUS:
				setHealthStatus((Double)newValue);
				return;
			case ComparchPackage.COMPONENT__PERFORMANCE_OLD:
				setPerformanceOld((Double)newValue);
				return;
			case ComparchPackage.COMPONENT__ORIGINAL_REQUEST:
				setOriginalRequest((Double)newValue);
				return;
			case ComparchPackage.COMPONENT__DEPLOYED_ON:
				setDeployedOn((Server)newValue);
				return;
			case ComparchPackage.COMPONENT__REPLICA:
				setReplica((Replica)newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case ComparchPackage.COMPONENT__PARAMETERS:
				getParameters().clear();
				return;
			case ComparchPackage.COMPONENT__REQUIRED_INTERFACES:
				getRequiredInterfaces().clear();
				return;
			case ComparchPackage.COMPONENT__STATE:
				setState(STATE_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT__PROVIDED_INTERFACES:
				getProvidedInterfaces().clear();
				return;
			case ComparchPackage.COMPONENT__TYPE:
				setType((ComponentType)null);
				return;
			case ComparchPackage.COMPONENT__MONITORED_PROPERTIES:
				getMonitoredProperties().clear();
				return;
			case ComparchPackage.COMPONENT__TENANT:
				setTenant((Tenant)null);
				return;
			case ComparchPackage.COMPONENT__ISSUES:
				getIssues().clear();
				return;
			case ComparchPackage.COMPONENT__IN_USE_REPLICA:
				setInUseReplica(IN_USE_REPLICA_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT__HEALTH_STATUS:
				setHealthStatus(HEALTH_STATUS_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT__PERFORMANCE_OLD:
				setPerformanceOld(PERFORMANCE_OLD_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT__ORIGINAL_REQUEST:
				setOriginalRequest(ORIGINAL_REQUEST_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT__DEPLOYED_ON:
				setDeployedOn((Server)null);
				return;
			case ComparchPackage.COMPONENT__REPLICA:
				setReplica((Replica)null);
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case ComparchPackage.COMPONENT__PARAMETERS:
				return parameters != null && !parameters.isEmpty();
			case ComparchPackage.COMPONENT__REQUIRED_INTERFACES:
				return requiredInterfaces != null && !requiredInterfaces.isEmpty();
			case ComparchPackage.COMPONENT__STATE:
				return state != STATE_EDEFAULT;
			case ComparchPackage.COMPONENT__PROVIDED_INTERFACES:
				return providedInterfaces != null && !providedInterfaces.isEmpty();
			case ComparchPackage.COMPONENT__TYPE:
				return type != null;
			case ComparchPackage.COMPONENT__MONITORED_PROPERTIES:
				return monitoredProperties != null && !monitoredProperties.isEmpty();
			case ComparchPackage.COMPONENT__TENANT:
				return getTenant() != null;
			case ComparchPackage.COMPONENT__ISSUES:
				return issues != null && !issues.isEmpty();
			case ComparchPackage.COMPONENT__IN_USE_REPLICA:
				return inUseReplica != IN_USE_REPLICA_EDEFAULT;
			case ComparchPackage.COMPONENT__HEALTH_STATUS:
				return healthStatus != HEALTH_STATUS_EDEFAULT;
			case ComparchPackage.COMPONENT__PERFORMANCE_OLD:
				return performanceOld != PERFORMANCE_OLD_EDEFAULT;
			case ComparchPackage.COMPONENT__ORIGINAL_REQUEST:
				return originalRequest != ORIGINAL_REQUEST_EDEFAULT;
			case ComparchPackage.COMPONENT__DEPLOYED_ON:
				return deployedOn != null;
			case ComparchPackage.COMPONENT__REPLICA:
				return replica != null;
		}
		return super.eIsSet(featureID);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		if (eIsProxy()) return super.toString();

		StringBuffer result = new StringBuffer(super.toString());
		result.append(" (state: ");
		result.append(state);
		result.append(", inUseReplica: ");
		result.append(inUseReplica);
		result.append(", healthStatus: ");
		result.append(healthStatus);
		result.append(", performanceOld: ");
		result.append(performanceOld);
		result.append(", originalRequest: ");
		result.append(originalRequest);
		result.append(')');
		return result.toString();
	}

} //ComponentImpl
