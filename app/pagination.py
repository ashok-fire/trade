"""Shared fixed-size history paging; no mutation or destructive truncation."""
from sqlalchemy import func, select

PAGE_SIZE = 10
LATEST_LIMIT = 5


def pagination(total: int, page: int = 1, per_page: int = PAGE_SIZE) -> dict:
    pages = max(1, (total+per_page-1)//per_page)
    page = max(1, min(page, pages))
    offset = (page-1)*per_page
    return {"page":page, "pages":pages, "per_page":per_page, "total":total, "offset":offset,
            "first":offset+1 if total else 0, "last":min(offset+per_page, total)}


def database_page(session, model, page=1):
    pager = pagination(session.scalar(select(func.count()).select_from(model)) or 0, page)
    rows = session.scalars(select(model).order_by(model.created_at.desc(), model.id.desc())
                           .offset(pager["offset"]).limit(pager["per_page"])).all()
    return rows, pager
