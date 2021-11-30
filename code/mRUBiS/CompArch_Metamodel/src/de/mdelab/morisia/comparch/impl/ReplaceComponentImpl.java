/**
 */
package de.mdelab.morisia.comparch.impl;

import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.ComponentType;
import de.mdelab.morisia.comparch.ReplaceComponent;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;
import org.eclipse.emf.ecore.impl.ENotificationImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Replace Component</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ReplaceComponentImpl#getAlternativeComponentType <em>Alternative Component Type</em>}</li>
 * </ul>
 *
 * @generated
 */
public class ReplaceComponentImpl extends RuleImpl implements ReplaceComponent {
	/**
	 * The cached value of the '{@link #getAlternativeComponentType() <em>Alternative Component Type</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getAlternativeComponentType()
	 * @generated
	 * @ordered
	 */
	protected ComponentType alternativeComponentType;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ReplaceComponentImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ComparchPackage.Literals.REPLACE_COMPONENT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComponentType getAlternativeComponentType() {
		if (alternativeComponentType != null && alternativeComponentType.eIsProxy()) {
			InternalEObject oldAlternativeComponentType = (InternalEObject)alternativeComponentType;
			alternativeComponentType = (ComponentType)eResolveProxy(oldAlternativeComponentType);
			if (alternativeComponentType != oldAlternativeComponentType) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, ComparchPackage.REPLACE_COMPONENT__ALTERNATIVE_COMPONENT_TYPE, oldAlternativeComponentType, alternativeComponentType));
			}
		}
		return alternativeComponentType;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ComponentType basicGetAlternativeComponentType() {
		return alternativeComponentType;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setAlternativeComponentType(ComponentType newAlternativeComponentType) {
		ComponentType oldAlternativeComponentType = alternativeComponentType;
		alternativeComponentType = newAlternativeComponentType;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.REPLACE_COMPONENT__ALTERNATIVE_COMPONENT_TYPE, oldAlternativeComponentType, alternativeComponentType));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case ComparchPackage.REPLACE_COMPONENT__ALTERNATIVE_COMPONENT_TYPE:
				if (resolve) return getAlternativeComponentType();
				return basicGetAlternativeComponentType();
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
			case ComparchPackage.REPLACE_COMPONENT__ALTERNATIVE_COMPONENT_TYPE:
				setAlternativeComponentType((ComponentType)newValue);
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
			case ComparchPackage.REPLACE_COMPONENT__ALTERNATIVE_COMPONENT_TYPE:
				setAlternativeComponentType((ComponentType)null);
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
			case ComparchPackage.REPLACE_COMPONENT__ALTERNATIVE_COMPONENT_TYPE:
				return alternativeComponentType != null;
		}
		return super.eIsSet(featureID);
	}

} //ReplaceComponentImpl
