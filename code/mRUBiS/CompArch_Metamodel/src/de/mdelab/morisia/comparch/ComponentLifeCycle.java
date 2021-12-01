/**
 */
package de.mdelab.morisia.comparch;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.eclipse.emf.common.util.Enumerator;

/**
 * <!-- begin-user-doc -->
 * A representation of the literals of the enumeration '<em><b>Component Life Cycle</b></em>',
 * and utility methods for working with them.
 * <!-- end-user-doc -->
 * @see de.mdelab.morisia.comparch.ComparchPackage#getComponentLifeCycle()
 * @model
 * @generated
 */
public enum ComponentLifeCycle implements Enumerator {
	/**
	 * The '<em><b>DEPLOYED</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #DEPLOYED_VALUE
	 * @generated
	 * @ordered
	 */
	DEPLOYED(0, "DEPLOYED", "DEPLOYED"),

	/**
	 * The '<em><b>STARTED</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #STARTED_VALUE
	 * @generated
	 * @ordered
	 */
	STARTED(1, "STARTED", "STARTED"),

	/**
	 * The '<em><b>UNDEPLOYED</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #UNDEPLOYED_VALUE
	 * @generated
	 * @ordered
	 */
	UNDEPLOYED(2, "UNDEPLOYED", "UNDEPLOYED"),

	/**
	 * The '<em><b>NOT SUPPORTED</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #NOT_SUPPORTED_VALUE
	 * @generated
	 * @ordered
	 */
	NOT_SUPPORTED(3, "NOT_SUPPORTED", "NOT_SUPPORTED"), /**
	 * The '<em><b>REMOVED</b></em>' literal object.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @see #REMOVED_VALUE
	 * @generated
	 * @ordered
	 */
	REMOVED(0, "REMOVED", "REMOVED");

	/**
	 * The '<em><b>DEPLOYED</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>DEPLOYED</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #DEPLOYED
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int DEPLOYED_VALUE = 0;

	/**
	 * The '<em><b>STARTED</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>STARTED</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #STARTED
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int STARTED_VALUE = 1;

	/**
	 * The '<em><b>UNDEPLOYED</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>UNDEPLOYED</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #UNDEPLOYED
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int UNDEPLOYED_VALUE = 2;

	/**
	 * The '<em><b>NOT SUPPORTED</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>NOT SUPPORTED</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #NOT_SUPPORTED
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int NOT_SUPPORTED_VALUE = 3;

	/**
	 * The '<em><b>REMOVED</b></em>' literal value.
	 * <!-- begin-user-doc -->
	 * <p>
	 * If the meaning of '<em><b>REMOVED</b></em>' literal object isn't clear,
	 * there really should be more of a description here...
	 * </p>
	 * <!-- end-user-doc -->
	 * @see #REMOVED
	 * @model
	 * @generated
	 * @ordered
	 */
	public static final int REMOVED_VALUE = 0;

	/**
	 * An array of all the '<em><b>Component Life Cycle</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private static final ComponentLifeCycle[] VALUES_ARRAY =
		new ComponentLifeCycle[] {
			DEPLOYED,
			STARTED,
			UNDEPLOYED,
			NOT_SUPPORTED,
			REMOVED,
		};

	/**
	 * A public read-only list of all the '<em><b>Component Life Cycle</b></em>' enumerators.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public static final List<ComponentLifeCycle> VALUES = Collections.unmodifiableList(Arrays.asList(VALUES_ARRAY));

	/**
	 * Returns the '<em><b>Component Life Cycle</b></em>' literal with the specified literal value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param literal the literal.
	 * @return the matching enumerator or <code>null</code>.
	 * @generated
	 */
	public static ComponentLifeCycle get(String literal) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			ComponentLifeCycle result = VALUES_ARRAY[i];
			if (result.toString().equals(literal)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Component Life Cycle</b></em>' literal with the specified name.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param name the name.
	 * @return the matching enumerator or <code>null</code>.
	 * @generated
	 */
	public static ComponentLifeCycle getByName(String name) {
		for (int i = 0; i < VALUES_ARRAY.length; ++i) {
			ComponentLifeCycle result = VALUES_ARRAY[i];
			if (result.getName().equals(name)) {
				return result;
			}
		}
		return null;
	}

	/**
	 * Returns the '<em><b>Component Life Cycle</b></em>' literal with the specified integer value.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @param value the integer value.
	 * @return the matching enumerator or <code>null</code>.
	 * @generated
	 */
	public static ComponentLifeCycle get(int value) {
		switch (value) {
			case DEPLOYED_VALUE: return DEPLOYED;
			case STARTED_VALUE: return STARTED;
			case UNDEPLOYED_VALUE: return UNDEPLOYED;
			case NOT_SUPPORTED_VALUE: return NOT_SUPPORTED;
		}
		return null;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private final int value;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private final String name;

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private final String literal;

	/**
	 * Only this class can construct instances.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	private ComponentLifeCycle(int value, String name, String literal) {
		this.value = value;
		this.name = name;
		this.literal = literal;
	}

	/**
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	public int getValue() {
	  return value;
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
	public String getLiteral() {
	  return literal;
	}

	/**
	 * Returns the literal value of the enumerator, which is its string representation.
	 * <!-- begin-user-doc -->
	 * <!-- end-user-doc -->
	 * @generated
	 */
	@Override
	public String toString() {
		return literal;
	}
	
} //ComponentLifeCycle
