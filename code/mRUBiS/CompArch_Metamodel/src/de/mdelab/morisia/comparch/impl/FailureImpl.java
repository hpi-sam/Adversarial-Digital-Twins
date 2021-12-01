/**
 */
package de.mdelab.morisia.comparch.impl;

import de.mdelab.morisia.comparch.ComparchPackage;
import de.mdelab.morisia.comparch.Failure;
import de.mdelab.morisia.comparch.MethodSpecification;
import de.mdelab.morisia.comparch.ProvidedInterface;

import org.eclipse.emf.common.notify.Notification;
import org.eclipse.emf.common.notify.NotificationChain;

import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecore.InternalEObject;

import org.eclipse.emf.ecore.impl.ENotificationImpl;

import org.eclipse.emf.ecore.util.EcoreUtil;

/**
 * <!-- begin-user-doc -->
 * An implementation of the model object '<em><b>Failure</b></em>'.
 * <!-- end-user-doc -->
 * <p>
 * The following features are implemented:
 * </p>
 * <ul>
 *   <li>{@link de.mdelab.morisia.comparch.impl.FailureImpl#getName <em>Name</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.FailureImpl#getMessage <em>Message</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.FailureImpl#getInterface <em>Interface</em>}</li>
 *   <li>{@link de.mdelab.morisia.comparch.impl.FailureImpl#getMethod <em>Method</em>}</li>
 * </ul>
 *
 * @generated
 */
public class FailureImpl extends ArchitecturalElementImpl implements Failure {
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
	 * The default value of the '{@link #getMessage() <em>Message</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMessage()
	 * @generated
	 * @ordered
	 */
	protected static final String MESSAGE_EDEFAULT = null;

	/**
	 * The cached value of the '{@link #getMessage() <em>Message</em>}' attribute.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMessage()
	 * @generated
	 * @ordered
	 */
	protected String message = MESSAGE_EDEFAULT;

	/**
	 * The cached value of the '{@link #getMethod() <em>Method</em>}' reference.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #getMethod()
	 * @generated
	 * @ordered
	 */
	protected MethodSpecification method;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	protected FailureImpl() {
		super();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	protected EClass eStaticClass() {
		return ComparchPackage.Literals.FAILURE;
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
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.FAILURE__NAME, oldName, name));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public String getMessage() {
		return message;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setMessage(String newMessage) {
		String oldMessage = message;
		message = newMessage;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.FAILURE__MESSAGE, oldMessage, message));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public ProvidedInterface getInterface() {
		if (eContainerFeatureID() != ComparchPackage.FAILURE__INTERFACE) return null;
		return (ProvidedInterface)eInternalContainer();
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public NotificationChain basicSetInterface(ProvidedInterface newInterface, NotificationChain msgs) {
		msgs = eBasicSetContainer((InternalEObject)newInterface, ComparchPackage.FAILURE__INTERFACE, msgs);
		return msgs;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setInterface(ProvidedInterface newInterface) {
		if (newInterface != eInternalContainer() || (eContainerFeatureID() != ComparchPackage.FAILURE__INTERFACE && newInterface != null)) {
			if (EcoreUtil.isAncestor(this, newInterface))
				throw new IllegalArgumentException("Recursive containment not allowed for " + toString());
			NotificationChain msgs = null;
			if (eInternalContainer() != null)
				msgs = eBasicRemoveFromContainer(msgs);
			if (newInterface != null)
				msgs = ((InternalEObject)newInterface).eInverseAdd(this, ComparchPackage.PROVIDED_INTERFACE__FAILURES, ProvidedInterface.class, msgs);
			msgs = basicSetInterface(newInterface, msgs);
			if (msgs != null) msgs.dispatch();
		}
		else if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.FAILURE__INTERFACE, newInterface, newInterface));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MethodSpecification getMethod() {
		if (method != null && method.eIsProxy()) {
			InternalEObject oldMethod = (InternalEObject)method;
			method = (MethodSpecification)eResolveProxy(oldMethod);
			if (method != oldMethod) {
				if (eNotificationRequired())
					eNotify(new ENotificationImpl(this, Notification.RESOLVE, ComparchPackage.FAILURE__METHOD, oldMethod, method));
			}
		}
		return method;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public MethodSpecification basicGetMethod() {
		return method;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public void setMethod(MethodSpecification newMethod) {
		MethodSpecification oldMethod = method;
		method = newMethod;
		if (eNotificationRequired())
			eNotify(new ENotificationImpl(this, Notification.SET, ComparchPackage.FAILURE__METHOD, oldMethod, method));
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public NotificationChain eInverseAdd(InternalEObject otherEnd, int featureID, NotificationChain msgs) {
		switch (featureID) {
			case ComparchPackage.FAILURE__INTERFACE:
				if (eInternalContainer() != null)
					msgs = eBasicRemoveFromContainer(msgs);
				return basicSetInterface((ProvidedInterface)otherEnd, msgs);
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
			case ComparchPackage.FAILURE__INTERFACE:
				return basicSetInterface(null, msgs);
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
			case ComparchPackage.FAILURE__INTERFACE:
				return eInternalContainer().eInverseRemove(this, ComparchPackage.PROVIDED_INTERFACE__FAILURES, ProvidedInterface.class, msgs);
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
			case ComparchPackage.FAILURE__NAME:
				return getName();
			case ComparchPackage.FAILURE__MESSAGE:
				return getMessage();
			case ComparchPackage.FAILURE__INTERFACE:
				return getInterface();
			case ComparchPackage.FAILURE__METHOD:
				if (resolve) return getMethod();
				return basicGetMethod();
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
			case ComparchPackage.FAILURE__NAME:
				setName((String)newValue);
				return;
			case ComparchPackage.FAILURE__MESSAGE:
				setMessage((String)newValue);
				return;
			case ComparchPackage.FAILURE__INTERFACE:
				setInterface((ProvidedInterface)newValue);
				return;
			case ComparchPackage.FAILURE__METHOD:
				setMethod((MethodSpecification)newValue);
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
			case ComparchPackage.FAILURE__NAME:
				setName(NAME_EDEFAULT);
				return;
			case ComparchPackage.FAILURE__MESSAGE:
				setMessage(MESSAGE_EDEFAULT);
				return;
			case ComparchPackage.FAILURE__INTERFACE:
				setInterface((ProvidedInterface)null);
				return;
			case ComparchPackage.FAILURE__METHOD:
				setMethod((MethodSpecification)null);
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
			case ComparchPackage.FAILURE__NAME:
				return NAME_EDEFAULT == null ? name != null : !NAME_EDEFAULT.equals(name);
			case ComparchPackage.FAILURE__MESSAGE:
				return MESSAGE_EDEFAULT == null ? message != null : !MESSAGE_EDEFAULT.equals(message);
			case ComparchPackage.FAILURE__INTERFACE:
				return getInterface() != null;
			case ComparchPackage.FAILURE__METHOD:
				return method != null;
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
		result.append(", message: ");
		result.append(message);
		result.append(')');
		return result.toString();
	}

} //FailureImpl
