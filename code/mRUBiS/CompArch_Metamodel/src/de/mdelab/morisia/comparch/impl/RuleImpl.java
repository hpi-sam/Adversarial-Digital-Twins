/**
 */
package de.mdelab.morisia.comparch.impl;

import de.mdelab.morisia.comparch.Annotations;
import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Issue;
import de.mdelab.morisia.comparch.Rule;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

import org.eclipse.emf.ecore.util.EcoreUtil;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Rule</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.impl.RuleImpl#getCosts <em>Costs</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.RuleImpl#getAnnotations <em>Annotations</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.RuleImpl#getHandles <em>Handles</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.RuleImpl#getUtilityIncrease <em>Utility Increase</em>}</li>
 * </ul>
 *
 * @generated
 */
public abstract class RuleImpl extends EObjectImpl implements Rule {
	/**
	 * The default value of the '{@link #getCosts() <em>Costs</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCosts()
	 * @generated
	 * @ordered
	 */
	protected static final double COSTS_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getCosts() <em>Costs</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCosts()
	 * @generated
	 * @ordered
	 */
	protected double costs = COSTS_EDEFAULT;

	/**
	 * The cached value of the '{@link #getHandles() <em>Handles</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getHandles()
	 * @generated
	 * @ordered
	 */
	protected Issue handles;

	/**
	 * The default value of the '{@link #getUtilityIncrease() <em>Utility Increase</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUtilityIncrease()
	 * @generated
	 * @ordered
	 */
	protected static final double UTILITY_INCREASE_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getUtilityIncrease() <em>Utility Increase</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUtilityIncrease()
	 * @generated
	 * @ordered
	 */
	protected double utilityIncrease = UTILITY_INCREASE_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected RuleImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ComparchPackage.Literals.RULE;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getCosts() {
		return costs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCosts(double newCosts) {
		double oldCosts = costs;
		costs = newCosts;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.RULE__COSTS, oldCosts, costs));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Annotations getAnnotations() {
		if (eContainerFeatureID() != ComparchPackage.RULE__ANNOTATIONS) return null;
		return (Annotations)eInternalContainer();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetAnnotations(Annotations newAnnotations, NotificationChain msgs) {
		msgs = eBasicSetContainer((InternalEObject)newAnnotations, ComparchPackage.RULE__ANNOTATIONS, msgs);
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setAnnotations(Annotations newAnnotations) {
		if (newAnnotations != eInternalContainer() || (eContainerFeatureID() != ComparchPackage.RULE__ANNOTATIONS && newAnnotations != null)) {
			if (EcoreUtil.isAncestor(this, newAnnotations))
				throw new IllegalArgumentException("Recursive containment not allowed for " + toString());
			NotificationChain msgs = null;
			if (eInternalContainer() != null)
				msgs = eBasicRemoveFromContainer(msgs);
			if (newAnnotations != null)
				msgs = ((InternalEObject)newAnnotations).eInverseAdd(this, ComparchPackage.ANNOTATIONS__RULES, Annotations.class, msgs);
			msgs = basicSetAnnotations(newAnnotations, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.RULE__ANNOTATIONS, newAnnotations, newAnnotations));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Issue getHandles() {
		if (handles != null && handles.eIsProxy()) {
			InternalEObject oldHandles = (InternalEObject)handles;
			handles = (Issue)eResolveProxy(oldHandles);
			if (handles != oldHandles) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, ComparchPackage.RULE__HANDLES, oldHandles, handles));
			}
		}
		return handles;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Issue basicGetHandles() {
		return handles;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetHandles(Issue newHandles, NotificationChain msgs) {
		Issue oldHandles = handles;
		handles = newHandles;
		if (eNotificationRequired()) {
			ENotificationImpl notification = new ENotificationImpl(this, Notification.SET, ComparchPackage.RULE__HANDLES, oldHandles, newHandles);
			if (msgs == null) msgs = notification; else msgs.add(notification);
		}
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setHandles(Issue newHandles) {
		if (newHandles != handles) {
			NotificationChain msgs = null;
			if (handles != null)
				msgs = ((InternalEObject)handles).eInverseRemove(this, ComparchPackage.ISSUE__HANDLED_BY, Issue.class, msgs);
			if (newHandles != null)
				msgs = ((InternalEObject)newHandles).eInverseAdd(this, ComparchPackage.ISSUE__HANDLED_BY, Issue.class, msgs);
			msgs = basicSetHandles(newHandles, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.RULE__HANDLES, newHandles, newHandles));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getUtilityIncrease() {
		return utilityIncrease;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setUtilityIncrease(double newUtilityIncrease) {
		double oldUtilityIncrease = utilityIncrease;
		utilityIncrease = newUtilityIncrease;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.RULE__UTILITY_INCREASE, oldUtilityIncrease, utilityIncrease));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseAdd(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case ComparchPackage.RULE__ANNOTATIONS:
				if (eInternalContainer() != null)
					msgs = eBasicRemoveFromContainer(msgs);
				return basicSetAnnotations((Annotations)otherEnd, msgs);
			case ComparchPackage.RULE__HANDLES:
				if (handles != null)
					msgs = ((InternalEObject)handles).eInverseRemove(this, ComparchPackage.ISSUE__HANDLED_BY, Issue.class, msgs);
				return basicSetHandles((Issue)otherEnd, msgs);
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
			case ComparchPackage.RULE__ANNOTATIONS:
				return basicSetAnnotations(null, msgs);
			case ComparchPackage.RULE__HANDLES:
				return basicSetHandles(null, msgs);
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
			case ComparchPackage.RULE__ANNOTATIONS:
				return eInternalContainer().eInverseRemove(this, ComparchPackage.ANNOTATIONS__RULES, Annotations.class, msgs);
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
			case ComparchPackage.RULE__COSTS:
				return getCosts();
			case ComparchPackage.RULE__ANNOTATIONS:
				return getAnnotations();
			case ComparchPackage.RULE__HANDLES:
				if (resolve) return getHandles();
				return basicGetHandles();
			case ComparchPackage.RULE__UTILITY_INCREASE:
				return getUtilityIncrease();
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
			case ComparchPackage.RULE__COSTS:
				setCosts((Double)newValue);
				return;
			case ComparchPackage.RULE__ANNOTATIONS:
				setAnnotations((Annotations)newValue);
				return;
			case ComparchPackage.RULE__HANDLES:
				setHandles((Issue)newValue);
				return;
			case ComparchPackage.RULE__UTILITY_INCREASE:
				setUtilityIncrease((Double)newValue);
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
			case ComparchPackage.RULE__COSTS:
				setCosts(COSTS_EDEFAULT);
				return;
			case ComparchPackage.RULE__ANNOTATIONS:
				setAnnotations((Annotations)null);
				return;
			case ComparchPackage.RULE__HANDLES:
				setHandles((Issue)null);
				return;
			case ComparchPackage.RULE__UTILITY_INCREASE:
				setUtilityIncrease(UTILITY_INCREASE_EDEFAULT);
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
			case ComparchPackage.RULE__COSTS:
				return costs != COSTS_EDEFAULT;
			case ComparchPackage.RULE__ANNOTATIONS:
				return getAnnotations() != null;
			case ComparchPackage.RULE__HANDLES:
				return handles != null;
			case ComparchPackage.RULE__UTILITY_INCREASE:
				return utilityIncrease != UTILITY_INCREASE_EDEFAULT;
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
		result.append(" (costs: ");
		result.append(costs);
		result.append(", utilityIncrease: ");
		result.append(utilityIncrease);
		result.append(')');
		return result.toString();
	}

} //RuleImpl
