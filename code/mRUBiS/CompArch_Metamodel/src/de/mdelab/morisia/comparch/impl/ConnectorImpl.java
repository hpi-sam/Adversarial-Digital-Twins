/**
 */
package de.mdelab.morisia.comparch.impl;

import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Connector;
import de.mdelab.morisia.comparch.ProvidedInterface;
import de.mdelab.morisia.comparch.RequiredInterface;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EcoreUtil;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Connector</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ConnectorImpl#isLazyAdaptation <em>Lazy Adaptation</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ConnectorImpl#getSource <em>Source</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ConnectorImpl#getTarget <em>Target</em>}</li>
 * </ul>
 *
 * @generated
 */
public class ConnectorImpl extends ArchitecturalElementImpl implements Connector {
	/**
	 * The default value of the '{@link #isLazyAdaptation() <em>Lazy Adaptation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isLazyAdaptation()
	 * @generated
	 * @ordered
	 */
	protected static final boolean LAZY_ADAPTATION_EDEFAULT = false;

	/**
	 * The cached value of the '{@link #isLazyAdaptation() <em>Lazy Adaptation</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #isLazyAdaptation()
	 * @generated
	 * @ordered
	 */
	protected boolean lazyAdaptation = LAZY_ADAPTATION_EDEFAULT;

	/**
	 * The cached value of the '{@link #getTarget() <em>Target</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getTarget()
	 * @generated
	 * @ordered
	 */
	protected ProvidedInterface target;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated NOT
	 */
	protected ConnectorImpl() {
		super();
		this.uid = EcoreUtil.generateUUID();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ComparchPackage.Literals.CONNECTOR;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public boolean isLazyAdaptation() {
		return lazyAdaptation;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setLazyAdaptation(boolean newLazyAdaptation) {
		boolean oldLazyAdaptation = lazyAdaptation;
		lazyAdaptation = newLazyAdaptation;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.CONNECTOR__LAZY_ADAPTATION, oldLazyAdaptation, lazyAdaptation));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public RequiredInterface getSource() {
		if (eContainerFeatureID() != ComparchPackage.CONNECTOR__SOURCE) return null;
		return (RequiredInterface)eInternalContainer();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetSource(RequiredInterface newSource, NotificationChain msgs) {
		msgs = eBasicSetContainer((InternalEObject)newSource, ComparchPackage.CONNECTOR__SOURCE, msgs);
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSource(RequiredInterface newSource) {
		if (newSource != eInternalContainer() || (eContainerFeatureID() != ComparchPackage.CONNECTOR__SOURCE && newSource != null)) {
			if (EcoreUtil.isAncestor(this, newSource))
				throw new IllegalArgumentException("Recursive containment not allowed for " + toString());
			NotificationChain msgs = null;
			if (eInternalContainer() != null)
				msgs = eBasicRemoveFromContainer(msgs);
			if (newSource != null)
				msgs = ((InternalEObject)newSource).eInverseAdd(this, ComparchPackage.REQUIRED_INTERFACE__CONNECTOR, RequiredInterface.class, msgs);
			msgs = basicSetSource(newSource, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.CONNECTOR__SOURCE, newSource, newSource));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProvidedInterface getTarget() {
		if (target != null && target.eIsProxy()) {
			InternalEObject oldTarget = (InternalEObject)target;
			target = (ProvidedInterface)eResolveProxy(oldTarget);
			if (target != oldTarget) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, ComparchPackage.CONNECTOR__TARGET, oldTarget, target));
			}
		}
		return target;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProvidedInterface basicGetTarget() {
		return target;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetTarget(ProvidedInterface newTarget, NotificationChain msgs) {
		ProvidedInterface oldTarget = target;
		target = newTarget;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, ComparchPackage.CONNECTOR__TARGET, oldTarget, newTarget);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setTarget(ProvidedInterface newTarget) {
		if (newTarget != target) {
			NotificationChain msgs = null;
			if (target != null)
				msgs = ((InternalEObject)target).eInverseRemove(this, ComparchPackage.PROVIDED_INTERFACE__CONNECTORS, ProvidedInterface.class, msgs);
			if (newTarget != null)
				msgs = ((InternalEObject)newTarget).eInverseAdd(this, ComparchPackage.PROVIDED_INTERFACE__CONNECTORS, ProvidedInterface.class, msgs);
			msgs = basicSetTarget(newTarget, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.CONNECTOR__TARGET, newTarget, newTarget));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseAdd(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case ComparchPackage.CONNECTOR__SOURCE:
				if (eInternalContainer() != null)
					msgs = eBasicRemoveFromContainer(msgs);
				return basicSetSource((RequiredInterface)otherEnd, msgs);
			case ComparchPackage.CONNECTOR__TARGET:
				if (target != null)
					msgs = ((InternalEObject)target).eInverseRemove(this, ComparchPackage.PROVIDED_INTERFACE__CONNECTORS, ProvidedInterface.class, msgs);
				return basicSetTarget((ProvidedInterface)otherEnd, msgs);
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
			case ComparchPackage.CONNECTOR__SOURCE:
				return basicSetSource(null, msgs);
			case ComparchPackage.CONNECTOR__TARGET:
				return basicSetTarget(null, msgs);
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
			case ComparchPackage.CONNECTOR__SOURCE:
				return eInternalContainer().eInverseRemove(this, ComparchPackage.REQUIRED_INTERFACE__CONNECTOR, RequiredInterface.class, msgs);
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
			case ComparchPackage.CONNECTOR__LAZY_ADAPTATION:
				return isLazyAdaptation();
			case ComparchPackage.CONNECTOR__SOURCE:
				return getSource();
			case ComparchPackage.CONNECTOR__TARGET:
				if (resolve) return getTarget();
				return basicGetTarget();
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
			case ComparchPackage.CONNECTOR__LAZY_ADAPTATION:
				setLazyAdaptation((Boolean)newValue);
				return;
			case ComparchPackage.CONNECTOR__SOURCE:
				setSource((RequiredInterface)newValue);
				return;
			case ComparchPackage.CONNECTOR__TARGET:
				setTarget((ProvidedInterface)newValue);
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
			case ComparchPackage.CONNECTOR__LAZY_ADAPTATION:
				setLazyAdaptation(LAZY_ADAPTATION_EDEFAULT);
				return;
			case ComparchPackage.CONNECTOR__SOURCE:
				setSource((RequiredInterface)null);
				return;
			case ComparchPackage.CONNECTOR__TARGET:
				setTarget((ProvidedInterface)null);
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
			case ComparchPackage.CONNECTOR__LAZY_ADAPTATION:
				return lazyAdaptation != LAZY_ADAPTATION_EDEFAULT;
			case ComparchPackage.CONNECTOR__SOURCE:
				return getSource() != null;
			case ComparchPackage.CONNECTOR__TARGET:
				return target != null;
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
		result.append(" (lazyAdaptation: ");
		result.append(lazyAdaptation);
		result.append(')');
		return result.toString();
	}

} //ConnectorImpl
