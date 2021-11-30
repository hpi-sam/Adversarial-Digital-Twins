/**
 */
package de.mdelab.morisia.comparch.impl;

import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.InterfaceType;
import de.mdelab.morisia.comparch.MethodSpecification;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EcoreUtil;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Method Specification</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.impl.MethodSpecificationImpl#getSignature <em>Signature</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.MethodSpecificationImpl#getInterfaceType <em>Interface Type</em>}</li>
 * </ul>
 *
 * @generated
 */
public class MethodSpecificationImpl extends ArchitecturalElementImpl implements MethodSpecification {
	/**
	 * The default value of the '{@link #getSignature() <em>Signature</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSignature()
	 * @generated
	 * @ordered
	 */
	protected static final String SIGNATURE_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getSignature() <em>Signature</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getSignature()
	 * @generated
	 * @ordered
	 */
	protected String signature = SIGNATURE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected MethodSpecificationImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ComparchPackage.Literals.METHOD_SPECIFICATION;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getSignature() {
		return signature;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setSignature(String newSignature) {
		String oldSignature = signature;
		signature = newSignature;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.METHOD_SPECIFICATION__SIGNATURE, oldSignature, signature));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public InterfaceType getInterfaceType() {
		if (eContainerFeatureID() != ComparchPackage.METHOD_SPECIFICATION__INTERFACE_TYPE) return null;
		return (InterfaceType)eInternalContainer();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetInterfaceType(InterfaceType newInterfaceType, NotificationChain msgs) {
		msgs = eBasicSetContainer((InternalEObject)newInterfaceType, ComparchPackage.METHOD_SPECIFICATION__INTERFACE_TYPE, msgs);
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setInterfaceType(InterfaceType newInterfaceType) {
		if (newInterfaceType != eInternalContainer() || (eContainerFeatureID() != ComparchPackage.METHOD_SPECIFICATION__INTERFACE_TYPE && newInterfaceType != null)) {
			if (EcoreUtil.isAncestor(this, newInterfaceType))
				throw new IllegalArgumentException("Recursive containment not allowed for " + toString());
			NotificationChain msgs = null;
			if (eInternalContainer() != null)
				msgs = eBasicRemoveFromContainer(msgs);
			if (newInterfaceType != null)
				msgs = ((InternalEObject)newInterfaceType).eInverseAdd(this, ComparchPackage.INTERFACE_TYPE__METHOD_SPECIFICATION, InterfaceType.class, msgs);
			msgs = basicSetInterfaceType(newInterfaceType, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.METHOD_SPECIFICATION__INTERFACE_TYPE, newInterfaceType, newInterfaceType));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseAdd(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case ComparchPackage.METHOD_SPECIFICATION__INTERFACE_TYPE:
				if (eInternalContainer() != null)
					msgs = eBasicRemoveFromContainer(msgs);
				return basicSetInterfaceType((InterfaceType)otherEnd, msgs);
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
			case ComparchPackage.METHOD_SPECIFICATION__INTERFACE_TYPE:
				return basicSetInterfaceType(null, msgs);
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
			case ComparchPackage.METHOD_SPECIFICATION__INTERFACE_TYPE:
				return eInternalContainer().eInverseRemove(this, ComparchPackage.INTERFACE_TYPE__METHOD_SPECIFICATION, InterfaceType.class, msgs);
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
			case ComparchPackage.METHOD_SPECIFICATION__SIGNATURE:
				return getSignature();
			case ComparchPackage.METHOD_SPECIFICATION__INTERFACE_TYPE:
				return getInterfaceType();
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
			case ComparchPackage.METHOD_SPECIFICATION__SIGNATURE:
				setSignature((String)newValue);
				return;
			case ComparchPackage.METHOD_SPECIFICATION__INTERFACE_TYPE:
				setInterfaceType((InterfaceType)newValue);
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
			case ComparchPackage.METHOD_SPECIFICATION__SIGNATURE:
				setSignature(SIGNATURE_EDEFAULT);
				return;
			case ComparchPackage.METHOD_SPECIFICATION__INTERFACE_TYPE:
				setInterfaceType((InterfaceType)null);
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
			case ComparchPackage.METHOD_SPECIFICATION__SIGNATURE:
				return SIGNATURE_EDEFAULT == null ? signature != null : !SIGNATURE_EDEFAULT.equals(signature);
			case ComparchPackage.METHOD_SPECIFICATION__INTERFACE_TYPE:
				return getInterfaceType() != null;
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
		result.append(" (signature: ");
		result.append(signature);
		result.append(')');
		return result.toString();
	}

} //MethodSpecificationImpl
