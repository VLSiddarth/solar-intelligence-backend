# ============================================================
# tests/unit/test_ontology.py
# Unit tests — GraphRAG ontology schema validation
# ============================================================

import pytest
from core.ontology.schema import OntologyValidator, validator
from shared.models.entities import EntityType, RelationshipType


class TestEntityValidation:

    def test_valid_person_entity(self):
        ok, errs = validator.validate_entity(EntityType.PERSON, {"name": "Ada Lovelace", "role": "Engineer"})
        assert ok is True
        assert errs == []

    def test_person_missing_required_name(self):
        ok, errs = validator.validate_entity(EntityType.PERSON, {"role": "Engineer"})
        assert ok is False
        assert any("name" in e for e in errs)

    def test_person_invalid_email(self):
        ok, errs = validator.validate_entity(EntityType.PERSON, {
            "name": "Alice", "email": "not-an-email"
        })
        assert ok is False
        assert any("email" in e for e in errs)

    def test_valid_organization(self):
        ok, errs = validator.validate_entity(EntityType.ORGANIZATION, {
            "name": "Anthropic", "size": "startup"
        })
        assert ok is True

    def test_organization_invalid_size_enum(self):
        ok, errs = validator.validate_entity(EntityType.ORGANIZATION, {
            "name": "TechCorp", "size": "mega"   # Not in enum
        })
        assert ok is False

    def test_valid_technology(self):
        ok, errs = validator.validate_entity(EntityType.TECHNOLOGY, {
            "name": "Apache Kafka", "version": "3.7", "category": "infra"
        })
        assert ok is True

    def test_technology_invalid_category(self):
        ok, errs = validator.validate_entity(EntityType.TECHNOLOGY, {
            "name": "Kafka", "category": "quantum"
        })
        assert ok is False

    def test_valid_project_with_required_status(self):
        ok, errs = validator.validate_entity(EntityType.PROJECT, {
            "name": "SI Architecture", "status": "active"
        })
        assert ok is True

    def test_project_invalid_status(self):
        ok, errs = validator.validate_entity(EntityType.PROJECT, {
            "name": "Project X", "status": "in_progress"  # Not in enum
        })
        assert ok is False

    def test_project_missing_status(self):
        ok, errs = validator.validate_entity(EntityType.PROJECT, {"name": "Project X"})
        assert ok is False
        assert any("status" in e for e in errs)

    def test_valid_metric(self):
        ok, errs = validator.validate_entity(EntityType.METRIC, {
            "name": "p99_latency", "value": 450.3, "unit": "ms"
        })
        assert ok is True

    def test_metric_missing_value(self):
        ok, errs = validator.validate_entity(EntityType.METRIC, {"name": "latency"})
        assert ok is False

    def test_unknown_entity_type_returns_false(self):
        ok, errs = validator.validate_entity("UnknownType", {"name": "test"})
        assert ok is False
        assert len(errs) > 0

    def test_all_11_entity_types_have_schemas(self):
        from core.ontology.schema import ENTITY_SCHEMAS
        assert len(ENTITY_SCHEMAS) == 11
        for et in EntityType:
            assert et in ENTITY_SCHEMAS, f"Missing schema for {et}"


class TestRelationshipValidation:

    def test_valid_has_role_relationship(self):
        ok, errs = validator.validate_relationship(
            RelationshipType.HAS_ROLE,
            EntityType.PERSON,
            EntityType.ORGANIZATION,
            {"title": "CTO"},
        )
        assert ok is True

    def test_has_role_wrong_from_type(self):
        ok, errs = validator.validate_relationship(
            RelationshipType.HAS_ROLE,
            EntityType.TECHNOLOGY,   # Technology can't have a role in Org
            EntityType.ORGANIZATION,
            {},
        )
        assert ok is False

    def test_authored_by_document_to_person(self):
        ok, errs = validator.validate_relationship(
            RelationshipType.AUTHORED_BY,
            EntityType.DOCUMENT,
            EntityType.PERSON,
            {},
        )
        assert ok is True

    def test_authored_by_wrong_direction(self):
        ok, errs = validator.validate_relationship(
            RelationshipType.AUTHORED_BY,
            EntityType.PERSON,    # Person cannot author a Document in this direction
            EntityType.DOCUMENT,
            {},
        )
        assert ok is False

    def test_uses_project_to_technology(self):
        ok, errs = validator.validate_relationship(
            RelationshipType.USES,
            EntityType.PROJECT,
            EntityType.TECHNOLOGY,
            {"purpose": "message queue"},
        )
        assert ok is True

    def test_governs_regulation_to_org(self):
        ok, errs = validator.validate_relationship(
            RelationshipType.GOVERNS,
            EntityType.REGULATION,
            EntityType.ORGANIZATION,
            {"effective": True},
        )
        assert ok is True

    def test_unknown_relationship_returns_false(self):
        ok, errs = validator.validate_relationship(
            "UNKNOWN_REL",
            EntityType.PERSON,
            EntityType.ORGANIZATION,
            {},
        )
        assert ok is False