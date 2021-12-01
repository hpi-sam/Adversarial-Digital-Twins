/**
 */
package de.mdelab.morisia.comparch.impl;

import de.mdelab.morisia.comparch.Annotations;
import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Component;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.Rule;

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
 * An implementation of the model object '<em><b>Issue</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.impl.IssueImpl#getUtilityDrop <em>Utility Drop</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.IssueImpl#getAffectedComponent <em>Affected Component</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.IssueImpl#getAnnotations <em>Annotations</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.IssueImpl#getHandledBy <em>Handled By</em>}</li>
 * </ul>
 *
 * @generated
 */
public abstract class IssueImpl extends EObjectImpl implements Issue {
	/**
	 * The default value of the '{@link #getUtilityDrop() <em>Utility Drop</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUtilityDrop()
	 * @generated
	 * @ordered
	 */
	protected static final double UTILITY_DROP_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getUtilityDrop() <em>Utility Drop</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUtilityDrop()
	 * @generated
	 * @ordered
	 */
	protected double utilityDrop = UTILITY_DROP_EDEFAULT;

	/**
	 * The cached value of the '{@link #getAffectedComponent() <em>Affected Component</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getAffectedComponent()
	 * @generated
	 * @ordered
	 */
	protected Component affectedComponent;

	/**
	 * The cached value of the '{@link #getHandledBy() <em>Handled By</em>}' reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getHandledBy()
	 * @generated
	 * @ordered
	 */
	protected EList<Rule> handledBy;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected IssueImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ComparchPackage.Literals.ISSUE;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getUtilityDrop() {
		return utilityDrop;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setUtilityDrop(double newUtilityDrop) {
		double oldUtilityDrop = utilityDrop;
		utilityDrop = newUtilityDrop;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.ISSUE__UTILITY_DROP, oldUtilityDrop, utilityDrop));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Component getAffectedComponent() {
		if (affectedComponent != null && affectedComponent.eIsProxy()) {
			InternalEObject oldAffectedComponent = (InternalEObject)affectedComponent;
			affectedComponent = (Component)eResolveProxy(oldAffectedComponent);
			if (affectedComponent != oldAffectedComponent) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, ComparchPackage.ISSUE__AFFECTED_COMPONENT, oldAffectedComponent, affectedComponent));
			}
		}
		return affectedComponent;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Component basicGetAffectedComponent() {
		return affectedComponent;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetAffectedComponent(Component newAffectedComponent, NotificationChain msgs) {
		Component oldAffectedComponent = affectedComponent;
		affectedComponent = newAffectedComponent;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, ComparchPackage.ISSUE__AFFECTED_COMPONENT, oldAffectedComponent, newAffectedComponent);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setAffectedComponent(Component newAffectedComponent) {
		if (newAffectedComponent != affectedComponent) {
			NotificationChain msgs = null;
			if (affectedComponent != null)
				msgs = ((InternalEObject)affectedComponent).eInverseRemove(this, ComparchPackage.COMPONENT__ISSUES, Component.class, msgs);
			if (newAffectedComponent != null)
				msgs = ((InternalEObject)newAffectedComponent).eInverseAdd(this, ComparchPackage.COMPONENT__ISSUES, Component.class, msgs);
			msgs = basicSetAffectedComponent(newAffectedComponent, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.ISSUE__AFFECTED_COMPONENT, newAffectedComponent, newAffectedComponent));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Annotations getAnnotations() {
		if (eContainerFeatureID() != ComparchPackage.ISSUE__ANNOTATIONS) return null;
		return (Annotations)eInternalContainer();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetAnnotations(Annotations newAnnotations, NotificationChain msgs) {
		msgs = eBasicSetContainer((InternalEObject)newAnnotations, ComparchPackage.ISSUE__ANNOTATIONS, msgs);
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setAnnotations(Annotations newAnnotations) {
		if (newAnnotations != eInternalContainer() || (eContainerFeatureID() != ComparchPackage.ISSUE__ANNOTATIONS && newAnnotations != null)) {
			if (EcoreUtil.isAncestor(this, newAnnotations))
				throw new IllegalArgumentException("Recursive containment not allowed for " + toString());
			NotificationChain msgs = null;
			if (eInternalContainer() != null)
				msgs = eBasicRemoveFromContainer(msgs);
			if (newAnnotations != null)
				msgs = ((InternalEObject)newAnnotations).eInverseAdd(this, ComparchPackage.ANNOTATIONS__ISSUES, Annotations.class, msgs);
			msgs = basicSetAnnotations(newAnnotations, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.ISSUE__ANNOTATIONS, newAnnotations, newAnnotations));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Rule> getHandledBy() {
		if (handledBy == null) {
			handledBy = new EObjectWithInverseResolvingEList<Rule>(Rule.class, this, ComparchPackage.ISSUE__HANDLED_BY, ComparchPackage.RULE__HANDLES);
		}
		return handledBy;
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
			case ComparchPackage.ISSUE__AFFECTED_COMPONENT:
				if (affectedComponent != null)
					msgs = ((InternalEObject)affectedComponent).eInverseRemove(this, ComparchPackage.COMPONENT__ISSUES, Component.class, msgs);
				return basicSetAffectedComponent((Component)otherEnd, msgs);
			case ComparchPackage.ISSUE__ANNOTATIONS:
				if (eInternalContainer() != null)
					msgs = eBasicRemoveFromContainer(msgs);
				return basicSetAnnotations((Annotations)otherEnd, msgs);
			case ComparchPackage.ISSUE__HANDLED_BY:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getHandledBy()).basicAdd(otherEnd, msgs);
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
			case ComparchPackage.ISSUE__AFFECTED_COMPONENT:
				return basicSetAffectedComponent(null, msgs);
			case ComparchPackage.ISSUE__ANNOTATIONS:
				return basicSetAnnotations(null, msgs);
			case ComparchPackage.ISSUE__HANDLED_BY:
				return ((InternalEList<?>)getHandledBy()).basicRemove(otherEnd, msgs);
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
			case ComparchPackage.ISSUE__ANNOTATIONS:
				return eInternalContainer().eInverseRemove(this, ComparchPackage.ANNOTATIONS__ISSUES, Annotations.class, msgs);
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
			case ComparchPackage.ISSUE__UTILITY_DROP:
				return getUtilityDrop();
			case ComparchPackage.ISSUE__AFFECTED_COMPONENT:
				if (resolve) return getAffectedComponent();
				return basicGetAffectedComponent();
			case ComparchPackage.ISSUE__ANNOTATIONS:
				return getAnnotations();
			case ComparchPackage.ISSUE__HANDLED_BY:
				return getHandledBy();
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
			case ComparchPackage.ISSUE__UTILITY_DROP:
				setUtilityDrop((Double)newValue);
				return;
			case ComparchPackage.ISSUE__AFFECTED_COMPONENT:
				setAffectedComponent((Component)newValue);
				return;
			case ComparchPackage.ISSUE__ANNOTATIONS:
				setAnnotations((Annotations)newValue);
				return;
			case ComparchPackage.ISSUE__HANDLED_BY:
				getHandledBy().clear();
				getHandledBy().addAll((Collection<? extends Rule>)newValue);
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
			case ComparchPackage.ISSUE__UTILITY_DROP:
				setUtilityDrop(UTILITY_DROP_EDEFAULT);
				return;
			case ComparchPackage.ISSUE__AFFECTED_COMPONENT:
				setAffectedComponent((Component)null);
				return;
			case ComparchPackage.ISSUE__ANNOTATIONS:
				setAnnotations((Annotations)null);
				return;
			case ComparchPackage.ISSUE__HANDLED_BY:
				getHandledBy().clear();
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
			case ComparchPackage.ISSUE__UTILITY_DROP:
				return utilityDrop != UTILITY_DROP_EDEFAULT;
			case ComparchPackage.ISSUE__AFFECTED_COMPONENT:
				return affectedComponent != null;
			case ComparchPackage.ISSUE__ANNOTATIONS:
				return getAnnotations() != null;
			case ComparchPackage.ISSUE__HANDLED_BY:
				return handledBy != null && !handledBy.isEmpty();
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
		result.append(" (utilityDrop: ");
		result.append(utilityDrop);
		result.append(')');
		return result.toString();
	}

} //IssueImpl
