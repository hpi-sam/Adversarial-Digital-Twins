/**
 */
package de.mdelab.morisia.comparch.impl;

import de.mdelab.morisia.comparch.Cloud;
import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Replica;
import de.mdelab.morisia.comparch.Server;

import java.util.Collection;
import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;
import org.eclipse.emf.common.util.EList;
import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;
import org.eclipse.emf.ecore.util.EObjectWithInverseResolvingEList;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Server</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getCloud <em>Cloud</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getName <em>Name</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getDeployedReplicas <em>Deployed Replicas</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getDeployedComponent <em>Deployed Component</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getCPUCapacity <em>CPU Capacity</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getMemoryCapacity <em>Memory Capacity</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getStorageCapacity <em>Storage Capacity</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getCPUConsumption <em>CPU Consumption</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getMemoryConsumption <em>Memory Consumption</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getStorageConsumption <em>Storage Consumption</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getCost <em>Cost</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getCpuFactor <em>Cpu Factor</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getMemoryFactor <em>Memory Factor</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ServerImpl#getStorageFactor <em>Storage Factor</em>}</li>
 * </ul>
 *
 * @generated
 */
public class ServerImpl extends ArchitecturalElementImpl implements Server {
	/**
	 * The default value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected static final String NAME_EDEFAULT = null;
	/**
	 * The cached value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected String name = NAME_EDEFAULT;
	/**
	 * The cached value of the '{@link #getDeployedReplicas() <em>Deployed Replicas</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDeployedReplicas()
	 * @generated
	 * @ordered
	 */
	protected EList<Replica> deployedReplicas;
	/**
	 * The cached value of the '{@link #getDeployedComponent() <em>Deployed Component</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getDeployedComponent()
	 * @generated
	 * @ordered
	 */
	protected EList<Component> deployedComponent;
	/**
	 * The default value of the '{@link #getCPUCapacity() <em>CPU Capacity</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCPUCapacity()
	 * @generated
	 * @ordered
	 */
	protected static final double CPU_CAPACITY_EDEFAULT = 0.0;
	/**
	 * The cached value of the '{@link #getCPUCapacity() <em>CPU Capacity</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCPUCapacity()
	 * @generated
	 * @ordered
	 */
	protected double cpuCapacity = CPU_CAPACITY_EDEFAULT;
	/**
	 * The default value of the '{@link #getMemoryCapacity() <em>Memory Capacity</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMemoryCapacity()
	 * @generated
	 * @ordered
	 */
	protected static final double MEMORY_CAPACITY_EDEFAULT = 0.0;
	/**
	 * The cached value of the '{@link #getMemoryCapacity() <em>Memory Capacity</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMemoryCapacity()
	 * @generated
	 * @ordered
	 */
	protected double memoryCapacity = MEMORY_CAPACITY_EDEFAULT;
	/**
	 * The default value of the '{@link #getStorageCapacity() <em>Storage Capacity</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getStorageCapacity()
	 * @generated
	 * @ordered
	 */
	protected static final double STORAGE_CAPACITY_EDEFAULT = 0.0;
	/**
	 * The cached value of the '{@link #getStorageCapacity() <em>Storage Capacity</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getStorageCapacity()
	 * @generated
	 * @ordered
	 */
	protected double storageCapacity = STORAGE_CAPACITY_EDEFAULT;
	/**
	 * The default value of the '{@link #getCPUConsumption() <em>CPU Consumption</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCPUConsumption()
	 * @generated
	 * @ordered
	 */
	protected static final double CPU_CONSUMPTION_EDEFAULT = 0.0;
	/**
	 * The cached value of the '{@link #getCPUConsumption() <em>CPU Consumption</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCPUConsumption()
	 * @generated
	 * @ordered
	 */
	protected double cpuConsumption = CPU_CONSUMPTION_EDEFAULT;
	/**
	 * The default value of the '{@link #getMemoryConsumption() <em>Memory Consumption</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMemoryConsumption()
	 * @generated
	 * @ordered
	 */
	protected static final double MEMORY_CONSUMPTION_EDEFAULT = 0.0;
	/**
	 * The cached value of the '{@link #getMemoryConsumption() <em>Memory Consumption</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMemoryConsumption()
	 * @generated
	 * @ordered
	 */
	protected double memoryConsumption = MEMORY_CONSUMPTION_EDEFAULT;
	/**
	 * The default value of the '{@link #getStorageConsumption() <em>Storage Consumption</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getStorageConsumption()
	 * @generated
	 * @ordered
	 */
	protected static final double STORAGE_CONSUMPTION_EDEFAULT = 0.0;
	/**
	 * The cached value of the '{@link #getStorageConsumption() <em>Storage Consumption</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getStorageConsumption()
	 * @generated
	 * @ordered
	 */
	protected double storageConsumption = STORAGE_CONSUMPTION_EDEFAULT;
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
	 * The default value of the '{@link #getCpuFactor() <em>Cpu Factor</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCpuFactor()
	 * @generated
	 * @ordered
	 */
	protected static final double CPU_FACTOR_EDEFAULT = 0.0;
	/**
	 * The cached value of the '{@link #getCpuFactor() <em>Cpu Factor</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCpuFactor()
	 * @generated
	 * @ordered
	 */
	protected double cpuFactor = CPU_FACTOR_EDEFAULT;
	/**
	 * The default value of the '{@link #getMemoryFactor() <em>Memory Factor</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMemoryFactor()
	 * @generated
	 * @ordered
	 */
	protected static final double MEMORY_FACTOR_EDEFAULT = 0.0;
	/**
	 * The cached value of the '{@link #getMemoryFactor() <em>Memory Factor</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMemoryFactor()
	 * @generated
	 * @ordered
	 */
	protected double memoryFactor = MEMORY_FACTOR_EDEFAULT;
	/**
	 * The default value of the '{@link #getStorageFactor() <em>Storage Factor</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getStorageFactor()
	 * @generated
	 * @ordered
	 */
	protected static final double STORAGE_FACTOR_EDEFAULT = 0.0;
	/**
	 * The cached value of the '{@link #getStorageFactor() <em>Storage Factor</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getStorageFactor()
	 * @generated
	 * @ordered
	 */
	protected double storageFactor = STORAGE_FACTOR_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ServerImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ComparchPackage.Literals.SERVER;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getCPUCapacity() {
		return cpuCapacity;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCPUCapacity(double newCPUCapacity) {
		double oldCPUCapacity = cpuCapacity;
		cpuCapacity = newCPUCapacity;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.SERVER__CPU_CAPACITY, oldCPUCapacity, cpuCapacity));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getMemoryCapacity() {
		return memoryCapacity;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setMemoryCapacity(double newMemoryCapacity) {
		double oldMemoryCapacity = memoryCapacity;
		memoryCapacity = newMemoryCapacity;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.SERVER__MEMORY_CAPACITY, oldMemoryCapacity, memoryCapacity));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getStorageCapacity() {
		return storageCapacity;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setStorageCapacity(double newStorageCapacity) {
		double oldStorageCapacity = storageCapacity;
		storageCapacity = newStorageCapacity;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.SERVER__STORAGE_CAPACITY, oldStorageCapacity, storageCapacity));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getCPUConsumption() {
		return cpuConsumption;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCPUConsumption(double newCPUConsumption) {
		double oldCPUConsumption = cpuConsumption;
		cpuConsumption = newCPUConsumption;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.SERVER__CPU_CONSUMPTION, oldCPUConsumption, cpuConsumption));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getMemoryConsumption() {
		return memoryConsumption;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setMemoryConsumption(double newMemoryConsumption) {
		double oldMemoryConsumption = memoryConsumption;
		memoryConsumption = newMemoryConsumption;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.SERVER__MEMORY_CONSUMPTION, oldMemoryConsumption, memoryConsumption));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getStorageConsumption() {
		return storageConsumption;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setStorageConsumption(double newStorageConsumption) {
		double oldStorageConsumption = storageConsumption;
		storageConsumption = newStorageConsumption;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.SERVER__STORAGE_CONSUMPTION, oldStorageConsumption, storageConsumption));
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
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.SERVER__COST, oldCost, cost));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getCpuFactor() {
		return cpuFactor;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCpuFactor(double newCpuFactor) {
		double oldCpuFactor = cpuFactor;
		cpuFactor = newCpuFactor;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.SERVER__CPU_FACTOR, oldCpuFactor, cpuFactor));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getMemoryFactor() {
		return memoryFactor;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setMemoryFactor(double newMemoryFactor) {
		double oldMemoryFactor = memoryFactor;
		memoryFactor = newMemoryFactor;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.SERVER__MEMORY_FACTOR, oldMemoryFactor, memoryFactor));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getStorageFactor() {
		return storageFactor;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setStorageFactor(double newStorageFactor) {
		double oldStorageFactor = storageFactor;
		storageFactor = newStorageFactor;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.SERVER__STORAGE_FACTOR, oldStorageFactor, storageFactor));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Cloud getCloud() {
		if (eContainerFeatureID() != ComparchPackage.SERVER__CLOUD) return null;
		return (Cloud)eInternalContainer();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetCloud(Cloud newCloud, NotificationChain msgs) {
		msgs = eBasicSetContainer((InternalEObject)newCloud, ComparchPackage.SERVER__CLOUD, msgs);
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCloud(Cloud newCloud) {
		if (newCloud != eInternalContainer() || (eContainerFeatureID() != ComparchPackage.SERVER__CLOUD && newCloud != null)) {
			if (EcoreUtil.isAncestor(this, newCloud))
				throw new IllegalArgumentException("Recursive containment not allowed for " + toString());
			NotificationChain msgs = null;
			if (eInternalContainer() != null)
				msgs = eBasicRemoveFromContainer(msgs);
			if (newCloud != null)
				msgs = ((InternalEObject)newCloud).eInverseAdd(this, ComparchPackage.CLOUD__SERVERS, Cloud.class, msgs);
			msgs = basicSetCloud(newCloud, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.SERVER__CLOUD, newCloud, newCloud));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getName() {
		return name;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setName(String newName) {
		String oldName = name;
		name = newName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.SERVER__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Replica> getDeployedReplicas() {
		if (deployedReplicas == null) {
			deployedReplicas = new EObjectWithInverseResolvingEList<Replica>(Replica.class, this, ComparchPackage.SERVER__DEPLOYED_REPLICAS, ComparchPackage.REPLICA__DEPLOYED_ON);
		}
		return deployedReplicas;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Component> getDeployedComponent() {
		if (deployedComponent == null) {
			deployedComponent = new EObjectWithInverseResolvingEList<Component>(Component.class, this, ComparchPackage.SERVER__DEPLOYED_COMPONENT, ComparchPackage.COMPONENT__DEPLOYED_ON);
		}
		return deployedComponent;
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
			case ComparchPackage.SERVER__CLOUD:
				if (eInternalContainer() != null)
					msgs = eBasicRemoveFromContainer(msgs);
				return basicSetCloud((Cloud)otherEnd, msgs);
			case ComparchPackage.SERVER__DEPLOYED_REPLICAS:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getDeployedReplicas()).basicAdd(otherEnd, msgs);
			case ComparchPackage.SERVER__DEPLOYED_COMPONENT:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getDeployedComponent()).basicAdd(otherEnd, msgs);
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
			case ComparchPackage.SERVER__CLOUD:
				return basicSetCloud(null, msgs);
			case ComparchPackage.SERVER__DEPLOYED_REPLICAS:
				return ((InternalEList<?>)getDeployedReplicas()).basicRemove(otherEnd, msgs);
			case ComparchPackage.SERVER__DEPLOYED_COMPONENT:
				return ((InternalEList<?>)getDeployedComponent()).basicRemove(otherEnd, msgs);
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
			case ComparchPackage.SERVER__CLOUD:
				return eInternalContainer().eInverseRemove(this, ComparchPackage.CLOUD__SERVERS, Cloud.class, msgs);
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
			case ComparchPackage.SERVER__CLOUD:
				return getCloud();
			case ComparchPackage.SERVER__NAME:
				return getName();
			case ComparchPackage.SERVER__DEPLOYED_REPLICAS:
				return getDeployedReplicas();
			case ComparchPackage.SERVER__DEPLOYED_COMPONENT:
				return getDeployedComponent();
			case ComparchPackage.SERVER__CPU_CAPACITY:
				return getCPUCapacity();
			case ComparchPackage.SERVER__MEMORY_CAPACITY:
				return getMemoryCapacity();
			case ComparchPackage.SERVER__STORAGE_CAPACITY:
				return getStorageCapacity();
			case ComparchPackage.SERVER__CPU_CONSUMPTION:
				return getCPUConsumption();
			case ComparchPackage.SERVER__MEMORY_CONSUMPTION:
				return getMemoryConsumption();
			case ComparchPackage.SERVER__STORAGE_CONSUMPTION:
				return getStorageConsumption();
			case ComparchPackage.SERVER__COST:
				return getCost();
			case ComparchPackage.SERVER__CPU_FACTOR:
				return getCpuFactor();
			case ComparchPackage.SERVER__MEMORY_FACTOR:
				return getMemoryFactor();
			case ComparchPackage.SERVER__STORAGE_FACTOR:
				return getStorageFactor();
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
			case ComparchPackage.SERVER__CLOUD:
				setCloud((Cloud)newValue);
				return;
			case ComparchPackage.SERVER__NAME:
				setName((String)newValue);
				return;
			case ComparchPackage.SERVER__DEPLOYED_REPLICAS:
				getDeployedReplicas().clear();
				getDeployedReplicas().addAll((Collection<? extends Replica>)newValue);
				return;
			case ComparchPackage.SERVER__DEPLOYED_COMPONENT:
				getDeployedComponent().clear();
				getDeployedComponent().addAll((Collection<? extends Component>)newValue);
				return;
			case ComparchPackage.SERVER__CPU_CAPACITY:
				setCPUCapacity((Double)newValue);
				return;
			case ComparchPackage.SERVER__MEMORY_CAPACITY:
				setMemoryCapacity((Double)newValue);
				return;
			case ComparchPackage.SERVER__STORAGE_CAPACITY:
				setStorageCapacity((Double)newValue);
				return;
			case ComparchPackage.SERVER__CPU_CONSUMPTION:
				setCPUConsumption((Double)newValue);
				return;
			case ComparchPackage.SERVER__MEMORY_CONSUMPTION:
				setMemoryConsumption((Double)newValue);
				return;
			case ComparchPackage.SERVER__STORAGE_CONSUMPTION:
				setStorageConsumption((Double)newValue);
				return;
			case ComparchPackage.SERVER__COST:
				setCost((Double)newValue);
				return;
			case ComparchPackage.SERVER__CPU_FACTOR:
				setCpuFactor((Double)newValue);
				return;
			case ComparchPackage.SERVER__MEMORY_FACTOR:
				setMemoryFactor((Double)newValue);
				return;
			case ComparchPackage.SERVER__STORAGE_FACTOR:
				setStorageFactor((Double)newValue);
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
			case ComparchPackage.SERVER__CLOUD:
				setCloud((Cloud)null);
				return;
			case ComparchPackage.SERVER__NAME:
				setName(NAME_EDEFAULT);
				return;
			case ComparchPackage.SERVER__DEPLOYED_REPLICAS:
				getDeployedReplicas().clear();
				return;
			case ComparchPackage.SERVER__DEPLOYED_COMPONENT:
				getDeployedComponent().clear();
				return;
			case ComparchPackage.SERVER__CPU_CAPACITY:
				setCPUCapacity(CPU_CAPACITY_EDEFAULT);
				return;
			case ComparchPackage.SERVER__MEMORY_CAPACITY:
				setMemoryCapacity(MEMORY_CAPACITY_EDEFAULT);
				return;
			case ComparchPackage.SERVER__STORAGE_CAPACITY:
				setStorageCapacity(STORAGE_CAPACITY_EDEFAULT);
				return;
			case ComparchPackage.SERVER__CPU_CONSUMPTION:
				setCPUConsumption(CPU_CONSUMPTION_EDEFAULT);
				return;
			case ComparchPackage.SERVER__MEMORY_CONSUMPTION:
				setMemoryConsumption(MEMORY_CONSUMPTION_EDEFAULT);
				return;
			case ComparchPackage.SERVER__STORAGE_CONSUMPTION:
				setStorageConsumption(STORAGE_CONSUMPTION_EDEFAULT);
				return;
			case ComparchPackage.SERVER__COST:
				setCost(COST_EDEFAULT);
				return;
			case ComparchPackage.SERVER__CPU_FACTOR:
				setCpuFactor(CPU_FACTOR_EDEFAULT);
				return;
			case ComparchPackage.SERVER__MEMORY_FACTOR:
				setMemoryFactor(MEMORY_FACTOR_EDEFAULT);
				return;
			case ComparchPackage.SERVER__STORAGE_FACTOR:
				setStorageFactor(STORAGE_FACTOR_EDEFAULT);
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
			case ComparchPackage.SERVER__CLOUD:
				return getCloud() != null;
			case ComparchPackage.SERVER__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case ComparchPackage.SERVER__DEPLOYED_REPLICAS:
				return deployedReplicas != null && !deployedReplicas.isEmpty();
			case ComparchPackage.SERVER__DEPLOYED_COMPONENT:
				return deployedComponent != null && !deployedComponent.isEmpty();
			case ComparchPackage.SERVER__CPU_CAPACITY:
				return cpuCapacity != CPU_CAPACITY_EDEFAULT;
			case ComparchPackage.SERVER__MEMORY_CAPACITY:
				return memoryCapacity != MEMORY_CAPACITY_EDEFAULT;
			case ComparchPackage.SERVER__STORAGE_CAPACITY:
				return storageCapacity != STORAGE_CAPACITY_EDEFAULT;
			case ComparchPackage.SERVER__CPU_CONSUMPTION:
				return cpuConsumption != CPU_CONSUMPTION_EDEFAULT;
			case ComparchPackage.SERVER__MEMORY_CONSUMPTION:
				return memoryConsumption != MEMORY_CONSUMPTION_EDEFAULT;
			case ComparchPackage.SERVER__STORAGE_CONSUMPTION:
				return storageConsumption != STORAGE_CONSUMPTION_EDEFAULT;
			case ComparchPackage.SERVER__COST:
				return cost != COST_EDEFAULT;
			case ComparchPackage.SERVER__CPU_FACTOR:
				return cpuFactor != CPU_FACTOR_EDEFAULT;
			case ComparchPackage.SERVER__MEMORY_FACTOR:
				return memoryFactor != MEMORY_FACTOR_EDEFAULT;
			case ComparchPackage.SERVER__STORAGE_FACTOR:
				return storageFactor != STORAGE_FACTOR_EDEFAULT;
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
		result.append(" (name: ");
		result.append(name);
		result.append(", CPUCapacity: ");
		result.append(cpuCapacity);
		result.append(", MemoryCapacity: ");
		result.append(memoryCapacity);
		result.append(", StorageCapacity: ");
		result.append(storageCapacity);
		result.append(", CPUConsumption: ");
		result.append(cpuConsumption);
		result.append(", MemoryConsumption: ");
		result.append(memoryConsumption);
		result.append(", StorageConsumption: ");
		result.append(storageConsumption);
		result.append(", Cost: ");
		result.append(cost);
		result.append(", cpuFactor: ");
		result.append(cpuFactor);
		result.append(", memoryFactor: ");
		result.append(memoryFactor);
		result.append(", storageFactor: ");
		result.append(storageFactor);
		result.append(')');
		return result.toString();
	}

} //ServerImpl
