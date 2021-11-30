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

import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.ComparchFactory;
import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.ComponentLifeCycle;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.InterfaceType;
import de.mdelab.morisia.comparch.Parameter;
import de.mdelab.morisia.comparch.ParameterType;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.Replica;
import de.mdelab.morisia.comparch.RequiredInterface;

/**
 * <!-- begin-user-doc --> An implementation of the model object '
 * <em><b>Component Type</b></em>'. <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getName <em>Name</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getArchitecture <em>Architecture</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getRequiredInterfaceTypes <em>Required Interface Types</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getProvidedInterfaceTypes <em>Provided Interface Types</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getParameterTypes <em>Parameter Types</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getInstances <em>Instances</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getReliability <em>Reliability</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getSatPoint <em>Sat Point</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getPerformanceMax <em>Performance Max</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getReplicaCost <em>Replica Cost</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getReplicaADT <em>Replica ADT</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getCpuFactor <em>Cpu Factor</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getMemoryFactor <em>Memory Factor</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ComponentTypeImpl#getStorageFactor <em>Storage Factor</em>}</li>
 * </ul>
 *
 * @generated
 */
public class ComponentTypeImpl extends ArchitecturalElementImpl implements
		ComponentType {
	/**
	 * The default value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected static final String NAME_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getName() <em>Name</em>}' attribute.
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @see #getName()
	 * @generated
	 * @ordered
	 */
	protected String name = NAME_EDEFAULT;

	/**
	 * The cached value of the '{@link #getRequiredInterfaceTypes() <em>Required Interface Types</em>}' reference list.
	 * <!-- begin-user-doc
	 * --> <!-- end-user-doc -->
	 * @see #getRequiredInterfaceTypes()
	 * @generated
	 * @ordered
	 */
	protected EList<InterfaceType> requiredInterfaceTypes;

	/**
	 * The cached value of the '{@link #getProvidedInterfaceTypes() <em>Provided Interface Types</em>}' reference list.
	 * <!-- begin-user-doc
	 * --> <!-- end-user-doc -->
	 * @see #getProvidedInterfaceTypes()
	 * @generated
	 * @ordered
	 */
	protected EList<InterfaceType> providedInterfaceTypes;

	/**
	 * The cached value of the '{@link #getParameterTypes()
	 * <em>Parameter Types</em>}' containment reference list. <!--
	 * begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @see #getParameterTypes()
	 * @generated
	 * @ordered
	 */
	protected EList<ParameterType> parameterTypes;

	/**
	 * The cached value of the '{@link #getInstances() <em>Instances</em>}' reference list.
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @see #getInstances()
	 * @generated
	 * @ordered
	 */
	protected EList<Component> instances;

	/**
	 * The default value of the '{@link #getReliability() <em>Reliability</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getReliability()
	 * @generated
	 * @ordered
	 */
	protected static final double RELIABILITY_EDEFAULT = 1.0;

	/**
	 * The cached value of the '{@link #getReliability() <em>Reliability</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getReliability()
	 * @generated
	 * @ordered
	 */
	protected double reliability = RELIABILITY_EDEFAULT;

	/**
	 * The default value of the '{@link #getSatPoint() <em>Sat Point</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSatPoint()
	 * @generated
	 * @ordered
	 */
	protected static final double SAT_POINT_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getSatPoint() <em>Sat Point</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSatPoint()
	 * @generated
	 * @ordered
	 */
	protected double satPoint = SAT_POINT_EDEFAULT;

	/**
	 * The default value of the '{@link #getPerformanceMax() <em>Performance Max</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPerformanceMax()
	 * @generated
	 * @ordered
	 */
	protected static final double PERFORMANCE_MAX_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getPerformanceMax() <em>Performance Max</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPerformanceMax()
	 * @generated
	 * @ordered
	 */
	protected double performanceMax = PERFORMANCE_MAX_EDEFAULT;

	/**
	 * The default value of the '{@link #getReplicaCost() <em>Replica Cost</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getReplicaCost()
	 * @generated
	 * @ordered
	 */
	protected static final double REPLICA_COST_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getReplicaCost() <em>Replica Cost</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getReplicaCost()
	 * @generated
	 * @ordered
	 */
	protected double replicaCost = REPLICA_COST_EDEFAULT;

	/**
	 * The default value of the '{@link #getReplicaADT() <em>Replica ADT</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getReplicaADT()
	 * @generated
	 * @ordered
	 */
	protected static final double REPLICA_ADT_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getReplicaADT() <em>Replica ADT</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getReplicaADT()
	 * @generated
	 * @ordered
	 */
	protected double replicaADT = REPLICA_ADT_EDEFAULT;

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
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	protected ComponentTypeImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ComparchPackage.Literals.COMPONENT_TYPE;
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	public String getName() {
		return name;
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	public void setName(String newName) {
		String oldName = name;
		name = newName;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT_TYPE__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	public Architecture getArchitecture() {
		if (eContainerFeatureID() != ComparchPackage.COMPONENT_TYPE__ARCHITECTURE) return null;
		return (Architecture)eInternalContainer();
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetArchitecture(Architecture newArchitecture,
			NotificationChain msgs) {
		msgs = eBasicSetContainer((InternalEObject)newArchitecture, ComparchPackage.COMPONENT_TYPE__ARCHITECTURE, msgs);
		return msgs;
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	public void setArchitecture(Architecture newArchitecture) {
		if (newArchitecture != eInternalContainer() || (eContainerFeatureID() != ComparchPackage.COMPONENT_TYPE__ARCHITECTURE && newArchitecture != null)) {
			if (EcoreUtil.isAncestor(this, newArchitecture))
				throw new IllegalArgumentException("Recursive containment not allowed for " + toString());
			NotificationChain msgs = null;
			if (eInternalContainer() != null)
				msgs = eBasicRemoveFromContainer(msgs);
			if (newArchitecture != null)
				msgs = ((InternalEObject)newArchitecture).eInverseAdd(this, ComparchPackage.ARCHITECTURE__COMPONENT_TYPES, Architecture.class, msgs);
			msgs = basicSetArchitecture(newArchitecture, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT_TYPE__ARCHITECTURE, newArchitecture, newArchitecture));
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	public EList<InterfaceType> getRequiredInterfaceTypes() {
		if (requiredInterfaceTypes == null) {
			requiredInterfaceTypes = new EObjectWithInverseResolvingEList.ManyInverse<InterfaceType>(InterfaceType.class, this, ComparchPackage.COMPONENT_TYPE__REQUIRED_INTERFACE_TYPES, ComparchPackage.INTERFACE_TYPE__REQUIRING_COMPONENT_TYPES);
		}
		return requiredInterfaceTypes;
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	public EList<InterfaceType> getProvidedInterfaceTypes() {
		if (providedInterfaceTypes == null) {
			providedInterfaceTypes = new EObjectWithInverseResolvingEList.ManyInverse<InterfaceType>(InterfaceType.class, this, ComparchPackage.COMPONENT_TYPE__PROVIDED_INTERFACE_TYPES, ComparchPackage.INTERFACE_TYPE__PROVIDING_COMPONENT_TYPES);
		}
		return providedInterfaceTypes;
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	public EList<ParameterType> getParameterTypes() {
		if (parameterTypes == null) {
			parameterTypes = new EObjectContainmentWithInverseEList<ParameterType>(ParameterType.class, this, ComparchPackage.COMPONENT_TYPE__PARAMETER_TYPES, ComparchPackage.PARAMETER_TYPE__COMPONENT_TYPE);
		}
		return parameterTypes;
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Component> getInstances() {
		if (instances == null) {
			instances = new EObjectWithInverseResolvingEList<Component>(Component.class, this, ComparchPackage.COMPONENT_TYPE__INSTANCES, ComparchPackage.COMPONENT__TYPE);
		}
		return instances;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getReliability() {
		return reliability;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setReliability(double newReliability) {
		double oldReliability = reliability;
		reliability = newReliability;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT_TYPE__RELIABILITY, oldReliability, reliability));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getSatPoint() {
		return satPoint;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSatPoint(double newSatPoint) {
		double oldSatPoint = satPoint;
		satPoint = newSatPoint;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT_TYPE__SAT_POINT, oldSatPoint, satPoint));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getPerformanceMax() {
		return performanceMax;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPerformanceMax(double newPerformanceMax) {
		double oldPerformanceMax = performanceMax;
		performanceMax = newPerformanceMax;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT_TYPE__PERFORMANCE_MAX, oldPerformanceMax, performanceMax));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getReplicaCost() {
		return replicaCost;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setReplicaCost(double newReplicaCost) {
		double oldReplicaCost = replicaCost;
		replicaCost = newReplicaCost;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT_TYPE__REPLICA_COST, oldReplicaCost, replicaCost));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getReplicaADT() {
		return replicaADT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setReplicaADT(double newReplicaADT) {
		double oldReplicaADT = replicaADT;
		replicaADT = newReplicaADT;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT_TYPE__REPLICA_ADT, oldReplicaADT, replicaADT));
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
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT_TYPE__CPU_FACTOR, oldCpuFactor, cpuFactor));
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
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT_TYPE__MEMORY_FACTOR, oldMemoryFactor, memoryFactor));
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
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.COMPONENT_TYPE__STORAGE_FACTOR, oldStorageFactor, storageFactor));
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * 
	 * @generated NOT
	 */
	public Component instantiate() {
		ComparchFactory factory = ComparchFactory.eINSTANCE;

		// create component
		Component component = factory.createComponent();
		// set type
		component.setType(this);
		// attributes
		// component.setUid("Component_" + this.getName()
		// + EcoreUtil.generateUUID());
		component.setState(ComponentLifeCycle.UNDEPLOYED);
		// parameters
		for (ParameterType parameterType : this.getParameterTypes()) {
			Parameter parameter = factory.createParameter();
			parameter.setType(parameterType);
			// parameter.setUid("Parameter_" + parameterType.getName()
			// + EcoreUtil.generateUUID());
			parameter.setValue(parameterType.getDefaultValue());
			parameter.setLazyAdaptation(false);
			component.getParameters().add(parameter);
		}
		// required interface
		for (InterfaceType interfaceType : this.getRequiredInterfaceTypes()) {
			RequiredInterface requiredInterface = factory
					.createRequiredInterface();
			requiredInterface.setType(interfaceType);
			// requiredInterface.setUid("RequiredInterface_"
			// + interfaceType.getFullyQualifiedName()
			// + EcoreUtil.generateUUID());
			component.getRequiredInterfaces().add(requiredInterface);
		}
		// provided interface
		for (InterfaceType interfaceType : this.getProvidedInterfaceTypes()) {
			ProvidedInterface providedInterface = factory
					.createProvidedInterface();
			providedInterface.setType(interfaceType);
			// providedInterface.setUid("ProvidedInterface_"
			// + interfaceType.getFullyQualifiedName()
			// + EcoreUtil.generateUUID());
			component.getProvidedInterfaces().add(providedInterface);
		}
		// architecture
		return component;
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public NotificationChain eInverseAdd(InternalEObject otherEnd,
			int featureID, NotificationChain msgs) {
		switch (featureID) {
			case ComparchPackage.COMPONENT_TYPE__ARCHITECTURE:
				if (eInternalContainer() != null)
					msgs = eBasicRemoveFromContainer(msgs);
				return basicSetArchitecture((Architecture)otherEnd, msgs);
			case ComparchPackage.COMPONENT_TYPE__REQUIRED_INTERFACE_TYPES:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getRequiredInterfaceTypes()).basicAdd(otherEnd, msgs);
			case ComparchPackage.COMPONENT_TYPE__PROVIDED_INTERFACE_TYPES:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getProvidedInterfaceTypes()).basicAdd(otherEnd, msgs);
			case ComparchPackage.COMPONENT_TYPE__PARAMETER_TYPES:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getParameterTypes()).basicAdd(otherEnd, msgs);
			case ComparchPackage.COMPONENT_TYPE__INSTANCES:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getInstances()).basicAdd(otherEnd, msgs);
		}
		return super.eInverseAdd(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseRemove(InternalEObject otherEnd,
			int featureID, NotificationChain msgs) {
		switch (featureID) {
			case ComparchPackage.COMPONENT_TYPE__ARCHITECTURE:
				return basicSetArchitecture(null, msgs);
			case ComparchPackage.COMPONENT_TYPE__REQUIRED_INTERFACE_TYPES:
				return ((InternalEList<?>)getRequiredInterfaceTypes()).basicRemove(otherEnd, msgs);
			case ComparchPackage.COMPONENT_TYPE__PROVIDED_INTERFACE_TYPES:
				return ((InternalEList<?>)getProvidedInterfaceTypes()).basicRemove(otherEnd, msgs);
			case ComparchPackage.COMPONENT_TYPE__PARAMETER_TYPES:
				return ((InternalEList<?>)getParameterTypes()).basicRemove(otherEnd, msgs);
			case ComparchPackage.COMPONENT_TYPE__INSTANCES:
				return ((InternalEList<?>)getInstances()).basicRemove(otherEnd, msgs);
		}
		return super.eInverseRemove(otherEnd, featureID, msgs);
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eBasicRemoveFromContainerFeature(
			NotificationChain msgs) {
		switch (eContainerFeatureID()) {
			case ComparchPackage.COMPONENT_TYPE__ARCHITECTURE:
				return eInternalContainer().eInverseRemove(this, ComparchPackage.ARCHITECTURE__COMPONENT_TYPES, Architecture.class, msgs);
		}
		return super.eBasicRemoveFromContainerFeature(msgs);
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case ComparchPackage.COMPONENT_TYPE__NAME:
				return getName();
			case ComparchPackage.COMPONENT_TYPE__ARCHITECTURE:
				return getArchitecture();
			case ComparchPackage.COMPONENT_TYPE__REQUIRED_INTERFACE_TYPES:
				return getRequiredInterfaceTypes();
			case ComparchPackage.COMPONENT_TYPE__PROVIDED_INTERFACE_TYPES:
				return getProvidedInterfaceTypes();
			case ComparchPackage.COMPONENT_TYPE__PARAMETER_TYPES:
				return getParameterTypes();
			case ComparchPackage.COMPONENT_TYPE__INSTANCES:
				return getInstances();
			case ComparchPackage.COMPONENT_TYPE__RELIABILITY:
				return getReliability();
			case ComparchPackage.COMPONENT_TYPE__SAT_POINT:
				return getSatPoint();
			case ComparchPackage.COMPONENT_TYPE__PERFORMANCE_MAX:
				return getPerformanceMax();
			case ComparchPackage.COMPONENT_TYPE__REPLICA_COST:
				return getReplicaCost();
			case ComparchPackage.COMPONENT_TYPE__REPLICA_ADT:
				return getReplicaADT();
			case ComparchPackage.COMPONENT_TYPE__CPU_FACTOR:
				return getCpuFactor();
			case ComparchPackage.COMPONENT_TYPE__MEMORY_FACTOR:
				return getMemoryFactor();
			case ComparchPackage.COMPONENT_TYPE__STORAGE_FACTOR:
				return getStorageFactor();
		}
		return super.eGet(featureID, resolve, coreType);
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	@SuppressWarnings("unchecked")
	@Override
	public void eSet(int featureID, Object newValue) {
		switch (featureID) {
			case ComparchPackage.COMPONENT_TYPE__NAME:
				setName((String)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__ARCHITECTURE:
				setArchitecture((Architecture)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__REQUIRED_INTERFACE_TYPES:
				getRequiredInterfaceTypes().clear();
				getRequiredInterfaceTypes().addAll((Collection<? extends InterfaceType>)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__PROVIDED_INTERFACE_TYPES:
				getProvidedInterfaceTypes().clear();
				getProvidedInterfaceTypes().addAll((Collection<? extends InterfaceType>)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__PARAMETER_TYPES:
				getParameterTypes().clear();
				getParameterTypes().addAll((Collection<? extends ParameterType>)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__INSTANCES:
				getInstances().clear();
				getInstances().addAll((Collection<? extends Component>)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__RELIABILITY:
				setReliability((Double)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__SAT_POINT:
				setSatPoint((Double)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__PERFORMANCE_MAX:
				setPerformanceMax((Double)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__REPLICA_COST:
				setReplicaCost((Double)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__REPLICA_ADT:
				setReplicaADT((Double)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__CPU_FACTOR:
				setCpuFactor((Double)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__MEMORY_FACTOR:
				setMemoryFactor((Double)newValue);
				return;
			case ComparchPackage.COMPONENT_TYPE__STORAGE_FACTOR:
				setStorageFactor((Double)newValue);
				return;
		}
		super.eSet(featureID, newValue);
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public void eUnset(int featureID) {
		switch (featureID) {
			case ComparchPackage.COMPONENT_TYPE__NAME:
				setName(NAME_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT_TYPE__ARCHITECTURE:
				setArchitecture((Architecture)null);
				return;
			case ComparchPackage.COMPONENT_TYPE__REQUIRED_INTERFACE_TYPES:
				getRequiredInterfaceTypes().clear();
				return;
			case ComparchPackage.COMPONENT_TYPE__PROVIDED_INTERFACE_TYPES:
				getProvidedInterfaceTypes().clear();
				return;
			case ComparchPackage.COMPONENT_TYPE__PARAMETER_TYPES:
				getParameterTypes().clear();
				return;
			case ComparchPackage.COMPONENT_TYPE__INSTANCES:
				getInstances().clear();
				return;
			case ComparchPackage.COMPONENT_TYPE__RELIABILITY:
				setReliability(RELIABILITY_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT_TYPE__SAT_POINT:
				setSatPoint(SAT_POINT_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT_TYPE__PERFORMANCE_MAX:
				setPerformanceMax(PERFORMANCE_MAX_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT_TYPE__REPLICA_COST:
				setReplicaCost(REPLICA_COST_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT_TYPE__REPLICA_ADT:
				setReplicaADT(REPLICA_ADT_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT_TYPE__CPU_FACTOR:
				setCpuFactor(CPU_FACTOR_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT_TYPE__MEMORY_FACTOR:
				setMemoryFactor(MEMORY_FACTOR_EDEFAULT);
				return;
			case ComparchPackage.COMPONENT_TYPE__STORAGE_FACTOR:
				setStorageFactor(STORAGE_FACTOR_EDEFAULT);
				return;
		}
		super.eUnset(featureID);
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public boolean eIsSet(int featureID) {
		switch (featureID) {
			case ComparchPackage.COMPONENT_TYPE__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case ComparchPackage.COMPONENT_TYPE__ARCHITECTURE:
				return getArchitecture() != null;
			case ComparchPackage.COMPONENT_TYPE__REQUIRED_INTERFACE_TYPES:
				return requiredInterfaceTypes != null && !requiredInterfaceTypes.isEmpty();
			case ComparchPackage.COMPONENT_TYPE__PROVIDED_INTERFACE_TYPES:
				return providedInterfaceTypes != null && !providedInterfaceTypes.isEmpty();
			case ComparchPackage.COMPONENT_TYPE__PARAMETER_TYPES:
				return parameterTypes != null && !parameterTypes.isEmpty();
			case ComparchPackage.COMPONENT_TYPE__INSTANCES:
				return instances != null && !instances.isEmpty();
			case ComparchPackage.COMPONENT_TYPE__RELIABILITY:
				return reliability != RELIABILITY_EDEFAULT;
			case ComparchPackage.COMPONENT_TYPE__SAT_POINT:
				return satPoint != SAT_POINT_EDEFAULT;
			case ComparchPackage.COMPONENT_TYPE__PERFORMANCE_MAX:
				return performanceMax != PERFORMANCE_MAX_EDEFAULT;
			case ComparchPackage.COMPONENT_TYPE__REPLICA_COST:
				return replicaCost != REPLICA_COST_EDEFAULT;
			case ComparchPackage.COMPONENT_TYPE__REPLICA_ADT:
				return replicaADT != REPLICA_ADT_EDEFAULT;
			case ComparchPackage.COMPONENT_TYPE__CPU_FACTOR:
				return cpuFactor != CPU_FACTOR_EDEFAULT;
			case ComparchPackage.COMPONENT_TYPE__MEMORY_FACTOR:
				return memoryFactor != MEMORY_FACTOR_EDEFAULT;
			case ComparchPackage.COMPONENT_TYPE__STORAGE_FACTOR:
				return storageFactor != STORAGE_FACTOR_EDEFAULT;
		}
		return super.eIsSet(featureID);
	}

	/**
	 * <!-- begin-user-doc --> <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		if (eIsProxy()) return super.toString();

		StringBuffer result = new StringBuffer(super.toString());
		result.append(" (name: ");
		result.append(name);
		result.append(", reliability: ");
		result.append(reliability);
		result.append(", satPoint: ");
		result.append(satPoint);
		result.append(", performanceMax: ");
		result.append(performanceMax);
		result.append(", replicaCost: ");
		result.append(replicaCost);
		result.append(", replicaADT: ");
		result.append(replicaADT);
		result.append(", cpuFactor: ");
		result.append(cpuFactor);
		result.append(", memoryFactor: ");
		result.append(memoryFactor);
		result.append(", storageFactor: ");
		result.append(storageFactor);
		result.append(')');
		return result.toString();
	}

} // ComponentTypeImpl
