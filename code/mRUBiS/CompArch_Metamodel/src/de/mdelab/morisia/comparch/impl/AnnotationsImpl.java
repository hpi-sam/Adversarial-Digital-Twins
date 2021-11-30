/**
 */
package de.mdelab.morisia.comparch.impl;

import de.mdelab.morisia.comparch.Annotations;
import de.mdelab.morisia.comparch.Architecture;
import de.mdelab.morisia.comparch.ComparchPackage;
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

import org.eclipse.emf.ecore.util.EObjectContainmentWithInverseEList;
import org.eclipse.emf.ecore.util.EcoreUtil;
import org.eclipse.emf.ecore.util.InternalEList;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Annotations</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.impl.AnnotationsImpl#getArchitecture <em>Architecture</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.AnnotationsImpl#getIssues <em>Issues</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.AnnotationsImpl#getRules <em>Rules</em>}</li>
 * </ul>
 *
 * @generated
 */
public class AnnotationsImpl extends EObjectImpl implements Annotations {
	/**
	 * The cached value of the '{@link #getIssues() <em>Issues</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getIssues()
	 * @generated
	 * @ordered
	 */
	protected EList<Issue> issues;

	/**
	 * The cached value of the '{@link #getRules() <em>Rules</em>}' containment reference list.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getRules()
	 * @generated
	 * @ordered
	 */
	protected EList<Rule> rules;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected AnnotationsImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ComparchPackage.Literals.ANNOTATIONS;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public Architecture getArchitecture() {
		if (eContainerFeatureID() != ComparchPackage.ANNOTATIONS__ARCHITECTURE) return null;
		return (Architecture)eInternalContainer();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetArchitecture(Architecture newArchitecture, NotificationChain msgs) {
		msgs = eBasicSetContainer((InternalEObject)newArchitecture, ComparchPackage.ANNOTATIONS__ARCHITECTURE, msgs);
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setArchitecture(Architecture newArchitecture) {
		if (newArchitecture != eInternalContainer() || (eContainerFeatureID() != ComparchPackage.ANNOTATIONS__ARCHITECTURE && newArchitecture != null)) {
			if (EcoreUtil.isAncestor(this, newArchitecture))
				throw new IllegalArgumentException("Recursive containment not allowed for " + toString());
			NotificationChain msgs = null;
			if (eInternalContainer() != null)
				msgs = eBasicRemoveFromContainer(msgs);
			if (newArchitecture != null)
				msgs = ((InternalEObject)newArchitecture).eInverseAdd(this, ComparchPackage.ARCHITECTURE__ANNOTATIONS, Architecture.class, msgs);
			msgs = basicSetArchitecture(newArchitecture, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.ANNOTATIONS__ARCHITECTURE, newArchitecture, newArchitecture));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Issue> getIssues() {
		if (issues == null) {
			issues = new EObjectContainmentWithInverseEList<Issue>(Issue.class, this, ComparchPackage.ANNOTATIONS__ISSUES, ComparchPackage.ISSUE__ANNOTATIONS);
		}
		return issues;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public EList<Rule> getRules() {
		if (rules == null) {
			rules = new EObjectContainmentWithInverseEList<Rule>(Rule.class, this, ComparchPackage.ANNOTATIONS__RULES, ComparchPackage.RULE__ANNOTATIONS);
		}
		return rules;
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
			case ComparchPackage.ANNOTATIONS__ARCHITECTURE:
				if (eInternalContainer() != null)
					msgs = eBasicRemoveFromContainer(msgs);
				return basicSetArchitecture((Architecture)otherEnd, msgs);
			case ComparchPackage.ANNOTATIONS__ISSUES:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getIssues()).basicAdd(otherEnd, msgs);
			case ComparchPackage.ANNOTATIONS__RULES:
				return ((InternalEList<InternalEObject>)(InternalEList<?>)getRules()).basicAdd(otherEnd, msgs);
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
			case ComparchPackage.ANNOTATIONS__ARCHITECTURE:
				return basicSetArchitecture(null, msgs);
			case ComparchPackage.ANNOTATIONS__ISSUES:
				return ((InternalEList<?>)getIssues()).basicRemove(otherEnd, msgs);
			case ComparchPackage.ANNOTATIONS__RULES:
				return ((InternalEList<?>)getRules()).basicRemove(otherEnd, msgs);
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
			case ComparchPackage.ANNOTATIONS__ARCHITECTURE:
				return eInternalContainer().eInverseRemove(this, ComparchPackage.ARCHITECTURE__ANNOTATIONS, Architecture.class, msgs);
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
			case ComparchPackage.ANNOTATIONS__ARCHITECTURE:
				return getArchitecture();
			case ComparchPackage.ANNOTATIONS__ISSUES:
				return getIssues();
			case ComparchPackage.ANNOTATIONS__RULES:
				return getRules();
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
			case ComparchPackage.ANNOTATIONS__ARCHITECTURE:
				setArchitecture((Architecture)newValue);
				return;
			case ComparchPackage.ANNOTATIONS__ISSUES:
				getIssues().clear();
				getIssues().addAll((Collection<? extends Issue>)newValue);
				return;
			case ComparchPackage.ANNOTATIONS__RULES:
				getRules().clear();
				getRules().addAll((Collection<? extends Rule>)newValue);
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
			case ComparchPackage.ANNOTATIONS__ARCHITECTURE:
				setArchitecture((Architecture)null);
				return;
			case ComparchPackage.ANNOTATIONS__ISSUES:
				getIssues().clear();
				return;
			case ComparchPackage.ANNOTATIONS__RULES:
				getRules().clear();
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
			case ComparchPackage.ANNOTATIONS__ARCHITECTURE:
				return getArchitecture() != null;
			case ComparchPackage.ANNOTATIONS__ISSUES:
				return issues != null && !issues.isEmpty();
			case ComparchPackage.ANNOTATIONS__RULES:
				return rules != null && !rules.isEmpty();
		}
		return super.eIsSet(featureID);
	}

} //AnnotationsImpl
