/**
 */
package de.mdelab.morisia.comparch.impl;

import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.Replica;

import de.mdelab.morisia.comparch.Server;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EcoreUtil;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Replica</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ReplicaImpl#getDeployedOn <em>Deployed On</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ReplicaImpl#getCost <em>Cost</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ReplicaImpl#getInUse <em>In Use</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ReplicaImpl#getCPURequirement <em>CPU Requirement</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ReplicaImpl#getMemoryRequirement <em>Memory Requirement</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ReplicaImpl#getStorageRequirement <em>Storage Requirement</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ReplicaImpl#getComponent <em>Component</em>}</li>
 * </ul>
 *
 * @generated
 */
public class ReplicaImpl extends ArchitecturalElementImpl implements Replica {
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
	 * The default value of the '{@link #getCost() <em>Cost</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCost()
	 * @generated
	 * @ordered
	 */
	protected static final double COST_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getCost() <em>Cost</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCost()
	 * @generated
	 * @ordered
	 */
	protected double cost = COST_EDEFAULT;

	/**
	 * The default value of the '{@link #getInUse() <em>In Use</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getInUse()
	 * @generated
	 * @ordered
	 */
	protected static final double IN_USE_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getInUse() <em>In Use</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getInUse()
	 * @generated
	 * @ordered
	 */
	protected double inUse = IN_USE_EDEFAULT;

	/**
	 * The default value of the '{@link #getCPURequirement() <em>CPU Requirement</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCPURequirement()
	 * @generated
	 * @ordered
	 */
	protected static final double CPU_REQUIREMENT_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getCPURequirement() <em>CPU Requirement</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCPURequirement()
	 * @generated
	 * @ordered
	 */
	protected double cpuRequirement = CPU_REQUIREMENT_EDEFAULT;

	/**
	 * The default value of the '{@link #getMemoryRequirement() <em>Memory Requirement</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMemoryRequirement()
	 * @generated
	 * @ordered
	 */
	protected static final double MEMORY_REQUIREMENT_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getMemoryRequirement() <em>Memory Requirement</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMemoryRequirement()
	 * @generated
	 * @ordered
	 */
	protected double memoryRequirement = MEMORY_REQUIREMENT_EDEFAULT;

	/**
	 * The default value of the '{@link #getStorageRequirement() <em>Storage Requirement</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getStorageRequirement()
	 * @generated
	 * @ordered
	 */
	protected static final double STORAGE_REQUIREMENT_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getStorageRequirement() <em>Storage Requirement</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getStorageRequirement()
	 * @generated
	 * @ordered
	 */
	protected double storageRequirement = STORAGE_REQUIREMENT_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ReplicaImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ComparchPackage.Literals.REPLICA;
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
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, ComparchPackage.REPLICA__DEPLOYED_ON, oldDeployedOn, deployedOn));
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
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, ComparchPackage.REPLICA__DEPLOYED_ON, oldDeployedOn, newDeployedOn);
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
				msgs = ((InternalEObject)deployedOn).eInverseRemove(this, ComparchPackage.SERVER__DEPLOYED_REPLICAS, Server.class, msgs);
			if (newDeployedOn != null)
				msgs = ((InternalEObject)newDeployedOn).eInverseAdd(this, ComparchPackage.SERVER__DEPLOYED_REPLICAS, Server.class, msgs);
			msgs = basicSetDeployedOn(newDeployedOn, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.REPLICA__DEPLOYED_ON, newDeployedOn, newDeployedOn));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getCost() {
		return cost;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCost(double newCost) {
		double oldCost = cost;
		cost = newCost;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.REPLICA__COST, oldCost, cost));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getInUse() {
		return inUse;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setInUse(double newInUse) {
		double oldInUse = inUse;
		inUse = newInUse;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.REPLICA__IN_USE, oldInUse, inUse));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getCPURequirement() {
		return cpuRequirement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCPURequirement(double newCPURequirement) {
		double oldCPURequirement = cpuRequirement;
		cpuRequirement = newCPURequirement;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.REPLICA__CPU_REQUIREMENT, oldCPURequirement, cpuRequirement));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getMemoryRequirement() {
		return memoryRequirement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setMemoryRequirement(double newMemoryRequirement) {
		double oldMemoryRequirement = memoryRequirement;
		memoryRequirement = newMemoryRequirement;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.REPLICA__MEMORY_REQUIREMENT, oldMemoryRequirement, memoryRequirement));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getStorageRequirement() {
		return storageRequirement;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setStorageRequirement(double newStorageRequirement) {
		double oldStorageRequirement = storageRequirement;
		storageRequirement = newStorageRequirement;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.REPLICA__STORAGE_REQUIREMENT, oldStorageRequirement, storageRequirement));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Component getComponent() {
		if (eContainerFeatureID() != ComparchPackage.REPLICA__COMPONENT) return null;
		return (Component)eInternalContainer();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetComponent(Component newComponent, NotificationChain msgs) {
		msgs = eBasicSetContainer((InternalEObject)newComponent, ComparchPackage.REPLICA__COMPONENT, msgs);
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setComponent(Component newComponent) {
		if (newComponent != eInternalContainer() || (eContainerFeatureID() != ComparchPackage.REPLICA__COMPONENT && newComponent != null)) {
			if (EcoreUtil.isAncestor(this, newComponent))
				throw new IllegalArgumentException("Recursive containment not allowed for " + toString());
			NotificationChain msgs = null;
			if (eInternalContainer() != null)
				msgs = eBasicRemoveFromContainer(msgs);
			if (newComponent != null)
				msgs = ((InternalEObject)newComponent).eInverseAdd(this, ComparchPackage.COMPONENT__REPLICA, Component.class, msgs);
			msgs = basicSetComponent(newComponent, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.REPLICA__COMPONENT, newComponent, newComponent));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseAdd(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case ComparchPackage.REPLICA__DEPLOYED_ON:
				if (deployedOn != null)
					msgs = ((InternalEObject)deployedOn).eInverseRemove(this, ComparchPackage.SERVER__DEPLOYED_REPLICAS, Server.class, msgs);
				return basicSetDeployedOn((Server)otherEnd, msgs);
			case ComparchPackage.REPLICA__COMPONENT:
				if (eInternalContainer() != null)
					msgs = eBasicRemoveFromContainer(msgs);
				return basicSetComponent((Component)otherEnd, msgs);
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
			case ComparchPackage.REPLICA__DEPLOYED_ON:
				return basicSetDeployedOn(null, msgs);
			case ComparchPackage.REPLICA__COMPONENT:
				return basicSetComponent(null, msgs);
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
			case ComparchPackage.REPLICA__COMPONENT:
				return eInternalContainer().eInverseRemove(this, ComparchPackage.COMPONENT__REPLICA, Component.class, msgs);
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
			case ComparchPackage.REPLICA__DEPLOYED_ON:
				if (resolve) return getDeployedOn();
				return basicGetDeployedOn();
			case ComparchPackage.REPLICA__COST:
				return getCost();
			case ComparchPackage.REPLICA__IN_USE:
				return getInUse();
			case ComparchPackage.REPLICA__CPU_REQUIREMENT:
				return getCPURequirement();
			case ComparchPackage.REPLICA__MEMORY_REQUIREMENT:
				return getMemoryRequirement();
			case ComparchPackage.REPLICA__STORAGE_REQUIREMENT:
				return getStorageRequirement();
			case ComparchPackage.REPLICA__COMPONENT:
				return getComponent();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case ComparchPackage.REPLICA__DEPLOYED_ON:
				setDeployedOn((Server)newValue);
				return;
			case ComparchPackage.REPLICA__COST:
				setCost((Double)newValue);
				return;
			case ComparchPackage.REPLICA__IN_USE:
				setInUse((Double)newValue);
				return;
			case ComparchPackage.REPLICA__CPU_REQUIREMENT:
				setCPURequirement((Double)newValue);
				return;
			case ComparchPackage.REPLICA__MEMORY_REQUIREMENT:
				setMemoryRequirement((Double)newValue);
				return;
			case ComparchPackage.REPLICA__STORAGE_REQUIREMENT:
				setStorageRequirement((Double)newValue);
				return;
			case ComparchPackage.REPLICA__COMPONENT:
				setComponent((Component)newValue);
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
			case ComparchPackage.REPLICA__DEPLOYED_ON:
				setDeployedOn((Server)null);
				return;
			case ComparchPackage.REPLICA__COST:
				setCost(COST_EDEFAULT);
				return;
			case ComparchPackage.REPLICA__IN_USE:
				setInUse(IN_USE_EDEFAULT);
				return;
			case ComparchPackage.REPLICA__CPU_REQUIREMENT:
				setCPURequirement(CPU_REQUIREMENT_EDEFAULT);
				return;
			case ComparchPackage.REPLICA__MEMORY_REQUIREMENT:
				setMemoryRequirement(MEMORY_REQUIREMENT_EDEFAULT);
				return;
			case ComparchPackage.REPLICA__STORAGE_REQUIREMENT:
				setStorageRequirement(STORAGE_REQUIREMENT_EDEFAULT);
				return;
			case ComparchPackage.REPLICA__COMPONENT:
				setComponent((Component)null);
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
			case ComparchPackage.REPLICA__DEPLOYED_ON:
				return deployedOn != null;
			case ComparchPackage.REPLICA__COST:
				return cost != COST_EDEFAULT;
			case ComparchPackage.REPLICA__IN_USE:
				return inUse != IN_USE_EDEFAULT;
			case ComparchPackage.REPLICA__CPU_REQUIREMENT:
				return cpuRequirement != CPU_REQUIREMENT_EDEFAULT;
			case ComparchPackage.REPLICA__MEMORY_REQUIREMENT:
				return memoryRequirement != MEMORY_REQUIREMENT_EDEFAULT;
			case ComparchPackage.REPLICA__STORAGE_REQUIREMENT:
				return storageRequirement != STORAGE_REQUIREMENT_EDEFAULT;
			case ComparchPackage.REPLICA__COMPONENT:
				return getComponent() != null;
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
		result.append(" (cost: ");
		result.append(cost);
		result.append(", inUse: ");
		result.append(inUse);
		result.append(", CPURequirement: ");
		result.append(cpuRequirement);
		result.append(", MemoryRequirement: ");
		result.append(memoryRequirement);
		result.append(", StorageRequirement: ");
		result.append(storageRequirement);
		result.append(')');
		return result.toString();
	}

} //ReplicaImpl
