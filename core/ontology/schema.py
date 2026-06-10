from typing import Any, Optional
from shared.models.entities import EntityType, RelationshipType
from shared.utils.logging import get_logger

logger = get_logger(__name__)

def _validate_str(v, nullable=False):
    if v is None: return nullable
    return isinstance(v, str) and len(v.strip()) > 0

def _validate_float_range(v, min_v, max_v, nullable=False):
    if v is None: return nullable
    try: return min_v <= float(v) <= max_v
    except: return False

def _validate_enum(v, choices, nullable=False):
    if v is None: return nullable
    return v in choices

def _validate_int(v, nullable=False):
    if v is None: return nullable
    return isinstance(v, int) and v >= 0

ENTITY_SCHEMAS = {
    EntityType.PERSON: {
        "required": ["name"],
        "fields": {"name": {"type":"str","nullable":False}, "role": {"type":"str","nullable":True}, "email": {"type":"str","nullable":True}},
        "validators": {"name": lambda v: _validate_str(v), "role": lambda v: _validate_str(v,True), "email": lambda v: v is None or ("@" in str(v) and "." in str(v))},
    },
    EntityType.ORGANIZATION: {
        "required": ["name"],
        "fields": {"name": {"type":"str","nullable":False}, "domain": {"type":"str","nullable":True}, "size": {"type":"enum","nullable":True,"choices":["startup","mid","enterprise"]}},
        "validators": {"name": lambda v: _validate_str(v), "domain": lambda v: _validate_str(v,True), "size": lambda v: _validate_enum(v,["startup","mid","enterprise"],True)},
    },
    EntityType.TECHNOLOGY: {
        "required": ["name"],
        "fields": {"name": {"type":"str","nullable":False}, "version": {"type":"str","nullable":True}, "category": {"type":"enum","nullable":True,"choices":["db","ml","infra","api","framework","tool"]}},
        "validators": {"name": lambda v: _validate_str(v), "version": lambda v: _validate_str(v,True), "category": lambda v: _validate_enum(v,["db","ml","infra","api","framework","tool"],True)},
    },
    EntityType.PROJECT: {
        "required": ["name","status"],
        "fields": {"name": {"type":"str","nullable":False}, "status": {"type":"enum","nullable":False,"choices":["active","archived","planned"]}, "owner_id": {"type":"str","nullable":True}},
        "validators": {"name": lambda v: _validate_str(v), "status": lambda v: _validate_enum(v,["active","archived","planned"]), "owner_id": lambda v: _validate_str(v,True)},
    },
    EntityType.DOCUMENT: {
        "required": ["title","ingested_at"],
        "fields": {"title": {"type":"str","nullable":False}, "source_url": {"type":"str","nullable":True}, "ingested_at": {"type":"datetime","nullable":False}, "chunk_count": {"type":"int","nullable":True}},
        "validators": {"title": lambda v: _validate_str(v), "source_url": lambda v: _validate_str(v,True), "ingested_at": lambda v: v is not None, "chunk_count": lambda v: _validate_int(v,True)},
    },
    EntityType.CONCEPT: {
        "required": ["name"],
        "fields": {"name": {"type":"str","nullable":False}, "domain": {"type":"str","nullable":True}, "definition": {"type":"str","nullable":True}},
        "validators": {"name": lambda v: _validate_str(v), "domain": lambda v: _validate_str(v,True), "definition": lambda v: _validate_str(v,True)},
    },
    EntityType.EVENT: {
        "required": ["name"],
        "fields": {"name": {"type":"str","nullable":False}, "date": {"type":"date","nullable":True}, "location": {"type":"str","nullable":True}},
        "validators": {"name": lambda v: _validate_str(v), "date": lambda v: v is None or isinstance(v,str), "location": lambda v: _validate_str(v,True)},
    },
    EntityType.METRIC: {
        "required": ["name","value"],
        "fields": {"name": {"type":"str","nullable":False}, "value": {"type":"float","nullable":False}, "unit": {"type":"str","nullable":True}, "measured_at": {"type":"datetime","nullable":True}},
        "validators": {"name": lambda v: _validate_str(v), "value": lambda v: isinstance(v,(int,float)), "unit": lambda v: _validate_str(v,True), "measured_at": lambda v: v is None or isinstance(v,str)},
    },
    EntityType.LOCATION: {
        "required": ["name"],
        "fields": {"name": {"type":"str","nullable":False}, "country": {"type":"str","nullable":True}, "coordinates": {"type":"str","nullable":True}},
        "validators": {"name": lambda v: _validate_str(v), "country": lambda v: _validate_str(v,True), "coordinates": lambda v: _validate_str(v,True)},
    },
    EntityType.PRODUCT: {
        "required": ["name"],
        "fields": {"name": {"type":"str","nullable":False}, "version": {"type":"str","nullable":True}, "vendor_id": {"type":"str","nullable":True}},
        "validators": {"name": lambda v: _validate_str(v), "version": lambda v: _validate_str(v,True), "vendor_id": lambda v: _validate_str(v,True)},
    },
    EntityType.REGULATION: {
        "required": ["name"],
        "fields": {"name": {"type":"str","nullable":False}, "jurisdiction": {"type":"str","nullable":True}, "effective_date": {"type":"date","nullable":True}},
        "validators": {"name": lambda v: _validate_str(v), "jurisdiction": lambda v: _validate_str(v,True), "effective_date": lambda v: v is None or isinstance(v,str)},
    },
}

RELATIONSHIP_CONSTRAINTS = {
    RelationshipType.HAS_ROLE:    {"allowed_from":[EntityType.PERSON],      "allowed_to":[EntityType.ORGANIZATION], "fields":{"title":{"type":"str","nullable":True},"since":{"type":"str","nullable":True}}},
    RelationshipType.AUTHORED_BY: {"allowed_from":[EntityType.DOCUMENT],    "allowed_to":[EntityType.PERSON],       "fields":{"date":{"type":"str","nullable":True}}},
    RelationshipType.RELATES_TO:  {"allowed_from":[EntityType.CONCEPT,EntityType.TECHNOLOGY,EntityType.PROJECT], "allowed_to":[EntityType.CONCEPT,EntityType.TECHNOLOGY,EntityType.PROJECT], "fields":{"strength":{"type":"float:0-1","nullable":True}}},
    RelationshipType.USES:        {"allowed_from":[EntityType.PROJECT,EntityType.PERSON,EntityType.ORGANIZATION], "allowed_to":[EntityType.TECHNOLOGY,EntityType.PRODUCT], "fields":{"purpose":{"type":"str","nullable":True}}},
    RelationshipType.BELONGS_TO:  {"allowed_from":[EntityType.PROJECT,EntityType.PERSON], "allowed_to":[EntityType.ORGANIZATION], "fields":{}},
    RelationshipType.MENTIONS:    {"allowed_from":[EntityType.DOCUMENT], "allowed_to":[EntityType.CONCEPT,EntityType.PERSON,EntityType.TECHNOLOGY,EntityType.ORGANIZATION,EntityType.EVENT], "fields":{"frequency":{"type":"int","nullable":True}}},
    RelationshipType.GOVERNS:     {"allowed_from":[EntityType.REGULATION], "allowed_to":[EntityType.ORGANIZATION,EntityType.TECHNOLOGY,EntityType.PROJECT], "fields":{"effective":{"type":"bool","nullable":True}}},
    RelationshipType.MEASURES:    {"allowed_from":[EntityType.METRIC], "allowed_to":[EntityType.PROJECT,EntityType.TECHNOLOGY,EntityType.ORGANIZATION], "fields":{}},
}

class OntologyValidator:
    @staticmethod
    def validate_entity(entity_type, fields):
        schema = ENTITY_SCHEMAS.get(entity_type)
        if not schema:
            return False, [f"Unknown entity type: {entity_type}"]
        errors = []
        for req_field in schema["required"]:
            if req_field not in fields or fields[req_field] is None:
                errors.append(f"Required field missing: {req_field}")
        for field_name, validator_fn in schema.get("validators", {}).items():
            if field_name in fields:
                try:
                    if not validator_fn(fields[field_name]):
                        errors.append(f"Field validation failed: {field_name} = {fields[field_name]!r}")
                except Exception as e:
                    errors.append(f"Field validator error on {field_name}: {e}")
        return len(errors) == 0, errors

    @staticmethod
    def validate_relationship(rel_type, from_type, to_type, fields):
        constraint = RELATIONSHIP_CONSTRAINTS.get(rel_type)
        if not constraint:
            return False, [f"Unknown relationship type: {rel_type}"]
        errors = []
        if from_type not in constraint["allowed_from"]:
            errors.append(f"{rel_type} cannot originate from {from_type}. Allowed: {constraint['allowed_from']}")
        if to_type not in constraint["allowed_to"]:
            errors.append(f"{rel_type} cannot target {to_type}. Allowed: {constraint['allowed_to']}")
        return len(errors) == 0, errors

validator = OntologyValidator()
