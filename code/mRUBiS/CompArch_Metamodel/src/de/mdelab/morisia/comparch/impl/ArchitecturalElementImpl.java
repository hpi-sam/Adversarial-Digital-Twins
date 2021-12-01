/**
 */
package de.mdelab.morisia.comparch.impl;

import de.mdelab.morisia.comparch.ArchitecturalElement;
import de.mdelab.morisia.comparch.ComparchPackage;

import org.eclipse.emf.common.notify.Notification;

import org.eclipse.emf.ecore.EClass;

import org.eclipse.emf.ecore.impl.ENotificationImpl;
import org.eclipse.emf.ecore.impl.EObjectImpl;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Architectural Element</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ArchitecturalElementImpl#getUid <em>Uid</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ArchitecturalElementImpl#getCriticality <em>Criticality</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ArchitecturalElementImpl#getImportance <em>Importance</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ArchitecturalElementImpl#getADT <em>ADT</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ArchitecturalElementImpl#getEfficiency <em>Efficiency</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ArchitecturalElementImpl#getPerformance <em>Performance</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.ArchitecturalElementImpl#getRequest <em>Request</em>}</li>
 * </ul>
 *
 * @generated
 */
public abstract class ArchitecturalElementImpl extends EObjectImpl implements ArchitecturalElement {
	/**
	 * The default value of the '{@link #getUid() <em>Uid</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUid()
	 * @generated
	 * @ordered
	 */
	protected static final String UID_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getUid() <em>Uid</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getUid()
	 * @generated
	 * @ordered
	 */
	protected String uid = UID_EDEFAULT;

	/**
	 * The default value of the '{@link #getCriticality() <em>Criticality</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCriticality()
	 * @generated
	 * @ordered
	 */
	protected static final double CRITICALITY_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getCriticality() <em>Criticality</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getCriticality()
	 * @generated
	 * @ordered
	 */
	protected double criticality = CRITICALITY_EDEFAULT;

	/**
	 * The default value of the '{@link #getImportance() <em>Importance</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getImportance()
	 * @generated
	 * @ordered
	 */
	protected static final double IMPORTANCE_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getImportance() <em>Importance</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getImportance()
	 * @generated
	 * @ordered
	 */
	protected double importance = IMPORTANCE_EDEFAULT;

	/**
	 * The default value of the '{@link #getADT() <em>ADT</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getADT()
	 * @generated
	 * @ordered
	 */
	protected static final double ADT_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getADT() <em>ADT</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getADT()
	 * @generated
	 * @ordered
	 */
	protected double aDT = ADT_EDEFAULT;

	/**
	 * The default value of the '{@link #getEfficiency() <em>Efficiency</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getEfficiency()
	 * @generated
	 * @ordered
	 */
	protected static final double EFFICIENCY_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getEfficiency() <em>Efficiency</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getEfficiency()
	 * @generated
	 * @ordered
	 */
	protected double efficiency = EFFICIENCY_EDEFAULT;

	/**
	 * The default value of the '{@link #getPerformance() <em>Performance</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPerformance()
	 * @generated
	 * @ordered
	 */
	protected static final double PERFORMANCE_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getPerformance() <em>Performance</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getPerformance()
	 * @generated
	 * @ordered
	 */
	protected double performance = PERFORMANCE_EDEFAULT;

	/**
	 * The default value of the '{@link #getRequest() <em>Request</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getRequest()
	 * @generated
	 * @ordered
	 */
	protected static final double REQUEST_EDEFAULT = 0.0;

	/**
	 * The cached value of the '{@link #getRequest() <em>Request</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getRequest()
	 * @generated
	 * @ordered
	 */
	protected double request = REQUEST_EDEFAULT;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected ArchitecturalElementImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ComparchPackage.Literals.ARCHITECTURAL_ELEMENT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getUid() {
		return uid;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setUid(String newUid) {
		String oldUid = uid;
		uid = newUid;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.ARCHITECTURAL_ELEMENT__UID, oldUid, uid));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getCriticality() {
		return criticality;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setCriticality(double newCriticality) {
		double oldCriticality = criticality;
		criticality = newCriticality;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.ARCHITECTURAL_ELEMENT__CRITICALITY, oldCriticality, criticality));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getImportance() {
		return importance;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setImportance(double newImportance) {
		double oldImportance = importance;
		importance = newImportance;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.ARCHITECTURAL_ELEMENT__IMPORTANCE, oldImportance, importance));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getADT() {
		return aDT;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setADT(double newADT) {
		double oldADT = aDT;
		aDT = newADT;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.ARCHITECTURAL_ELEMENT__ADT, oldADT, aDT));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getEfficiency() {
		return efficiency;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setEfficiency(double newEfficiency) {
		double oldEfficiency = efficiency;
		efficiency = newEfficiency;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.ARCHITECTURAL_ELEMENT__EFFICIENCY, oldEfficiency, efficiency));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getPerformance() {
		return performance;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setPerformance(double newPerformance) {
		double oldPerformance = performance;
		performance = newPerformance;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.ARCHITECTURAL_ELEMENT__PERFORMANCE, oldPerformance, performance));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public double getRequest() {
		return request;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setRequest(double newRequest) {
		double oldRequest = request;
		request = newRequest;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.ARCHITECTURAL_ELEMENT__REQUEST, oldRequest, request));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public Object eGet(int featureID, boolean resolve, boolean coreType) {
		switch (featureID) {
			case ComparchPackage.ARCHITECTURAL_ELEMENT__UID:
				return getUid();
			case ComparchPackage.ARCHITECTURAL_ELEMENT__CRITICALITY:
				return getCriticality();
			case ComparchPackage.ARCHITECTURAL_ELEMENT__IMPORTANCE:
				return getImportance();
			case ComparchPackage.ARCHITECTURAL_ELEMENT__ADT:
				return getADT();
			case ComparchPackage.ARCHITECTURAL_ELEMENT__EFFICIENCY:
				return getEfficiency();
			case ComparchPackage.ARCHITECTURAL_ELEMENT__PERFORMANCE:
				return getPerformance();
			case ComparchPackage.ARCHITECTURAL_ELEMENT__REQUEST:
				return getRequest();
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
			case ComparchPackage.ARCHITECTURAL_ELEMENT__UID:
				setUid((String)newValue);
				return;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__CRITICALITY:
				setCriticality((Double)newValue);
				return;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__IMPORTANCE:
				setImportance((Double)newValue);
				return;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__ADT:
				setADT((Double)newValue);
				return;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__EFFICIENCY:
				setEfficiency((Double)newValue);
				return;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__PERFORMANCE:
				setPerformance((Double)newValue);
				return;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__REQUEST:
				setRequest((Double)newValue);
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
			case ComparchPackage.ARCHITECTURAL_ELEMENT__UID:
				setUid(UID_EDEFAULT);
				return;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__CRITICALITY:
				setCriticality(CRITICALITY_EDEFAULT);
				return;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__IMPORTANCE:
				setImportance(IMPORTANCE_EDEFAULT);
				return;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__ADT:
				setADT(ADT_EDEFAULT);
				return;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__EFFICIENCY:
				setEfficiency(EFFICIENCY_EDEFAULT);
				return;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__PERFORMANCE:
				setPerformance(PERFORMANCE_EDEFAULT);
				return;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__REQUEST:
				setRequest(REQUEST_EDEFAULT);
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
			case ComparchPackage.ARCHITECTURAL_ELEMENT__UID:
				return UID_EDEFAULT == null ? uid != null : !UID_EDEFAULT.equals(uid);
			case ComparchPackage.ARCHITECTURAL_ELEMENT__CRITICALITY:
				return criticality != CRITICALITY_EDEFAULT;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__IMPORTANCE:
				return importance != IMPORTANCE_EDEFAULT;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__ADT:
				return aDT != ADT_EDEFAULT;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__EFFICIENCY:
				return efficiency != EFFICIENCY_EDEFAULT;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__PERFORMANCE:
				return performance != PERFORMANCE_EDEFAULT;
			case ComparchPackage.ARCHITECTURAL_ELEMENT__REQUEST:
				return request != REQUEST_EDEFAULT;
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
		result.append(" (uid: ");
		result.append(uid);
		result.append(", criticality: ");
		result.append(criticality);
		result.append(", importance: ");
		result.append(importance);
		result.append(", aDT: ");
		result.append(aDT);
		result.append(", efficiency: ");
		result.append(efficiency);
		result.append(", performance: ");
		result.append(performance);
		result.append(", request: ");
		result.append(request);
		result.append(')');
		return result.toString();
	}

} //ArchitecturalElementImpl
